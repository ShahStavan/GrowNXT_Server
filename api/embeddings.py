"""REST trigger and status for the Nifty 50 batch embedding job.

Two routes on a FastAPI router that `app.py` includes:

* ``POST /api/embeddings/nifty50/run`` launches `ingestion.batch.run_batch`
  in a **detached process** and returns ``202`` at once. Docling extraction and
  ``SentenceTransformer.encode`` are CPU-bound, GIL-holding calls; running them
  as a background task inside the serving process would stall every other
  request for hours, so the work never runs in-process. Starting hours of CPU
  is a more sensitive action than generating one report, so the route is gated
  by ``NIFTY50_ADMIN_TOKEN`` and disabled (``503``) when that is unset.
* ``GET /api/embeddings/nifty50/status[/{run_id}]`` reads the run manifest --
  a file read with no network or subprocess involved, safe to poll.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import OUTPUT_DIR
from ingestion import batch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embeddings/nifty50", tags=["embeddings"])


class EmbeddingRunRequest(BaseModel):
    """Body of ``POST /api/embeddings/nifty50/run``; mirrors the CLI flags."""

    tickers: list[str] = Field(default_factory=list)
    force: list[str] = Field(default_factory=list)
    force_all: bool = False
    annual_reports: int = Field(default=1, ge=0, le=10)
    transcripts: int = Field(default=1, ge=0, le=20)
    presentations: int = Field(default=1, ge=0, le=20)
    concall_years: int = Field(default=1, ge=1, le=10)
    limit: int | None = Field(default=None, ge=0)
    dry_run: bool = False
    notify: bool = True
    notify_per_ticker: bool = True
    # Compute knobs. Omitted means "resolve from this host", which is the right
    # answer on a Space whose hardware the caller cannot see.
    device: str | None = None
    num_threads: int | None = Field(default=None, ge=1, le=128)
    embed_batch_size: int | None = Field(default=None, ge=1, le=1024)
    fast_tables: bool = False
    page_filter: bool = False
    # Resumption and the persistence gate, mirroring the CLI. `resume` takes a
    # run id, or true for the newest unfinished run.
    resume: str | bool = False
    allow_ephemeral: bool = False
    catalog_ttl_hours: float | None = Field(default=None, ge=0, le=8760)
    no_checkpoint: bool = False


def require_admin(authorization: str | None, x_admin_token: str | None) -> None:
    """Raises unless a valid admin token was presented.

    Accepts ``Authorization: Bearer <token>`` or ``X-Admin-Token: <token>``.
    An unset server token disables the route rather than opening it.
    """
    if not os.getenv(batch.ADMIN_TOKEN_ENV, "").strip():
        raise HTTPException(
            status_code=503,
            detail=f"{batch.ADMIN_TOKEN_ENV} is not configured on this server",
        )
    presented = x_admin_token
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not batch.admin_token_ok(presented):
        raise HTTPException(status_code=401, detail="invalid or missing admin token")


def output_dir() -> Path:
    """The artefact root the routes read and write; patched by tests."""
    return OUTPUT_DIR


def _apply_resume(
    req: EmbeddingRunRequest, options: batch.BatchOptions
) -> batch.BatchOptions:
    """Resolves `resume` on a request into options that continue that run.

    Args:
        req: The parsed request body.
        options: The options built from it.

    Returns:
        The options to launch, unchanged when no resume was asked for.

    Raises:
        BatchError: If there is nothing to resume, or a locked setting differs.
    """
    if not req.resume:
        return options
    run_id = req.resume if isinstance(req.resume, str) else ""
    if not run_id:
        for candidate in batch.list_run_ids(output_dir()):
            manifest = batch.load_manifest(run_id=candidate, output_dir=output_dir())
            if manifest is not None and manifest.status != batch.COMPLETED:
                run_id = candidate
                break
    if not run_id:
        raise batch.BatchError("no unfinished run to resume")
    manifest = batch.load_manifest(run_id=run_id, output_dir=output_dir())
    if manifest is None:
        raise batch.BatchError(f"no manifest recorded for run {run_id}")
    options.resume = run_id
    return batch.merge_resume_options(manifest.options, options)


@router.post("/run", status_code=202)
def run_embeddings(
    body: EmbeddingRunRequest | None = None,
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    """Launches the batch detached and returns its run id immediately."""
    require_admin(authorization, x_admin_token)
    req = body or EmbeddingRunRequest()
    options = batch.BatchOptions(
        tickers=[t.strip().upper() for t in req.tickers if t.strip()],
        force=[t.strip().upper() for t in req.force if t.strip()],
        force_all=req.force_all,
        annual_reports=req.annual_reports,
        transcripts=req.transcripts,
        presentations=req.presentations,
        concall_years=req.concall_years,
        limit=req.limit,
        dry_run=req.dry_run,
        notify=req.notify,
        notify_per_ticker=req.notify_per_ticker,
        output_dir=str(output_dir()),
        device=req.device,
        num_threads=req.num_threads,
        embed_batch_size=req.embed_batch_size,
        fast_tables=req.fast_tables,
        page_filter=req.page_filter,
        allow_ephemeral=req.allow_ephemeral,
        no_checkpoint=req.no_checkpoint,
        catalog_ttl_hours=(
            batch.DEFAULT_CATALOG_TTL_HOURS
            if req.catalog_ttl_hours is None
            else req.catalog_ttl_hours
        ),
    )
    try:
        options = _apply_resume(req, options)
    except batch.BatchError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    try:
        run_id, log_path, pid = batch.launch_background(options)
    except batch.BatchAlreadyRunning as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    return {
        "run_id": run_id,
        "pid": pid,
        "status": batch.RUNNING,
        "log": str(log_path),
        "status_url": f"{router.prefix}/status/{run_id}",
    }


@router.get("/status")
def latest_status():
    """Progress of the latest run."""
    return batch.status_report(output_dir=output_dir())


@router.get("/status/{run_id}")
def run_status(run_id: str):
    """Progress of one run by id."""
    data = batch.status_report(run_id=run_id, output_dir=output_dir())
    if data.get("status") == "UNKNOWN_RUN":
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return data
