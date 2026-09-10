"""Batch embedding of the Nifty 50 corpus with incremental, resumable state.

This is an orchestration layer. It does not change what a chunk is or how it is
embedded; it wires together the stages that already exist -- catalogue,
download, Docling extraction, chunking, Qdrant indexing -- and runs them
across every constituent with top-level state of its own.

Two layers decide what work a run does:

* **Layer 1, the catalogue diff.** Once per ticker, the catalogue is fetched
  and the newest annual report, transcript and presentation are compared
  against ``output/<TICKER>/state.json``. Because a ``doc_id`` is derived from
  the filing's period (``annual_report_FY2026``), a newly published filing is
  one whose id has never been seen, and only those -- plus any that failed
  last time, or were produced by an older chunker/extractor, or live in a
  collection this configuration no longer writes to -- are downloaded,
  extracted and chunked at all.
* **Layer 2, the content fingerprint.** `QdrantVectorIndexer` compares each
  chunk set's fingerprint with what it recorded before embedding, so a filing
  re-issued under an existing id is still re-embedded, and an unchanged one
  never is.

In steady state a run is therefore fifty catalogue requests and fifty state
reads: no Docling, no embedding.

The batch is a plain function, `run_batch`, so the CLI, a detached background
process and an HTTP trigger are three ways to start the same work rather than
three implementations of it. A run records itself in one cross-ticker manifest
(``output/_nifty50/run_manifest.json``) that a status command or endpoint can
read without any network call, and logs every stage boundary through
`ingestion.runlog`.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from core.config import OUTPUT_DIR, PROJECT_ROOT, safe_ticker
from ingestion import stages
from ingestion.nifty50 import Constituent, load_nifty50, verified_as_of
from ingestion.runlog import RunLogger, new_run_id, setup_run_logging, utc_now

# Layer 1's policy -- which of the newest filings a run must process -- lives in
# `ingestion.stages`, beside the stages that act on its verdict. These names are
# re-exported because `batch.select_entries`, `batch.compute_pending`,
# `batch.pending_reason` and `batch.REASON_*` are the established call sites.
from ingestion.stages import (  # noqa: F401 - re-exported for existing callers
    REASON_COLLECTION,
    REASON_FORCED,
    REASON_NEW,
    REASON_RETRY,
    REASON_VERSION,
    compute_pending,
    pending_reason,
    select_entries,
)

if TYPE_CHECKING:
    from core.hardware import HardwareProfile
    from ingestion.indexer import IndexerConfig, QdrantVectorIndexer

logger = logging.getLogger(__name__)

JOB_NAME: str = "nifty50"
MANIFEST_DIR_NAME: str = "_nifty50"
MANIFEST_FILENAME: str = "run_manifest.json"
SCHEMA_VERSION: int = 1

# A RUNNING manifest older than this whose process cannot be confirmed alive
# is treated as abandoned so a crash never blocks the next run forever.
STALE_RUN_HOURS: int = 24

MAX_WORKERS: int = 4

# Below this much VRAM, concurrent tickers are refused: see `resolve_workers`.
# Docling's layout and table models plus the embedder are roughly a gigabyte
# resident per worker before a single page is rasterised.
GPU_CONCURRENCY_VRAM_GB: float = 8.0

DEFAULT_INTER_TICKER_DELAY: float = 1.5
DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100

# How long a ticker's catalogue check stays good for. Long enough that a
# restart the same day costs no requests at all, short enough that a filing
# published in the morning is picked up by an evening run. Overridden with
# ``--catalog-ttl`` or ``GROWNXT_CATALOG_TTL_HOURS``; zero disables the skip.
DEFAULT_CATALOG_TTL_HOURS: float = 6.0
CATALOG_TTL_ENV: str = "GROWNXT_CATALOG_TTL_HOURS"

# Run-level status.
RUNNING: str = "RUNNING"
COMPLETED: str = "COMPLETED"
COMPLETED_WITH_ERRORS: str = "COMPLETED_WITH_ERRORS"
FAILED: str = "FAILED"

# Per-ticker status.
TICKER_OK: str = "OK"
TICKER_PARTIAL: str = "PARTIAL"
TICKER_FAILED: str = "FAILED"
TICKER_NO_DOCUMENTS: str = "NO_DOCUMENTS"
TICKER_DRY_RUN: str = "DRY_RUN"

ADMIN_TOKEN_ENV: str = "NIFTY50_ADMIN_TOKEN"


class BatchError(RuntimeError):
    """Raised when a run cannot start."""


class BatchAlreadyRunning(BatchError):
    """Raised when a live run is already recorded in the manifest."""


# --- Options ---------------------------------------------------------------


@dataclass
class BatchOptions:
    """Everything that shapes one run; serialisable so a detached child can reload it.

    Attributes:
        tickers: Symbols in scope; empty means the whole Nifty 50 list.
        nifty50_file: Constituent CSV override.
        annual_reports: Newest annual reports to embed per ticker.
        transcripts: Newest concall transcripts to embed per ticker.
        presentations: Newest investor presentations to embed per ticker.
        concall_years: Years of concalls to request from the catalogue.
        force: Symbols whose known documents are re-processed regardless.
        force_all: Re-process every ticker in scope.
        limit: Process only the first N tickers (smoke testing).
        workers: Tickers processed concurrently, capped at `MAX_WORKERS`.
        dry_run: Catalogue and diff only; no download, extraction or embedding.
        run_id: Explicit identifier; generated when absent.
        notify: Whether to fire the webhook on completion.
        notify_per_ticker: Whether to also post one webhook message as
            each ticker finishes. Rides on the same `notify` switch and
            the same URL; off leaves only the end-of-run notification.
        webhook_url: Webhook override for this run.
        verbose: DEBUG-level per-run log.
        delay_seconds: Pause between tickers' catalogue requests.
        chunk_size: Chunker target size, in characters.
        chunk_overlap: Chunker overlap, in characters.
        output_dir: Artefact root; defaults to `OUTPUT_DIR`.
        log_dir: Log root; defaults to `ingestion.runlog.LOG_ROOT`.
        device: Where Docling's and the embedder's models run. None resolves
            through `core.hardware`.
        num_threads: CPU threads for extraction. None means physical cores.
        embed_batch_size: Texts per encode call. None sizes it from VRAM.
        fast_tables: Run TableFormer in fast rather than accurate mode. The
            dominant cost on a dense annual report, and the one setting here
            that changes the extracted text -- so it moves the recorded
            extractor version, and every document already indexed under the
            other mode is re-extracted on the next run.
        page_filter: Convert only the financial section of an annual report
            -- statements, notes, auditor's report and MD&A -- skipping the
            AGM notice, directors' report, governance and BRSR sections
            ahead of them. Roughly halves the pages Docling converts on a
            typical filing. Like `fast_tables` it changes the extracted
            text, so it moves the extractor version and anything indexed
            the other way is re-extracted. Documents under 60 pages, and
            any filing whose financial section cannot be located with
            confidence, are converted whole.
        catalog_ttl_hours: How long a ticker's last catalogue check stays good
            for. Inside the window, a ticker whose every filing is already
            embedded by this pipeline is skipped without the request. Zero
            disables it, which is what a run that must see today's filings
            wants.
        allow_ephemeral: Proceed even when the vector store is in-memory. Off,
            because a crash then loses every vector while `state.json` still
            records them as INDEXED -- the one failure mode where the next
            run's resumption is silently wrong rather than merely slow.
        resume: A run id whose checkpoint this run continues, reusing it as
            both the thread and the run id so the manifest and logs stay one
            logical run. None starts a new run.
        no_checkpoint: Run the plain sequential loop with no checkpointer.
            Reproduces the pre-graph behaviour exactly; the escape hatch if
            the saver misbehaves.
    """

    tickers: list[str] = field(default_factory=list)
    nifty50_file: str | None = None
    annual_reports: int = 1
    transcripts: int = 1
    presentations: int = 1
    concall_years: int = 1
    force: list[str] = field(default_factory=list)
    force_all: bool = False
    limit: int | None = None
    workers: int = 1
    dry_run: bool = False
    run_id: str | None = None
    notify: bool = True
    notify_per_ticker: bool = True
    webhook_url: str | None = None
    verbose: bool = False
    delay_seconds: float = DEFAULT_INTER_TICKER_DELAY
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    output_dir: str | None = None
    log_dir: str | None = None
    device: str | None = None
    num_threads: int | None = None
    embed_batch_size: int | None = None
    fast_tables: bool = False
    page_filter: bool = False
    catalog_ttl_hours: float = DEFAULT_CATALOG_TTL_HOURS
    allow_ephemeral: bool = False
    resume: str | None = None
    no_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchOptions:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).resolve() if self.output_dir else OUTPUT_DIR

    @property
    def log_root(self) -> Path | None:
        return Path(self.log_dir).resolve() if self.log_dir else None

    def forced(self, symbol: str) -> bool:
        """Whether Layer 1 is bypassed for a symbol."""
        if self.force_all:
            return True
        wanted = {safe_ticker(s) for s in self.force}
        return safe_ticker(symbol) in wanted

    def resolve_constituents(self) -> list[Constituent]:
        """Returns the constituents in scope, honouring `tickers` and `limit`."""
        path = Path(self.nifty50_file) if self.nifty50_file else None
        members = load_nifty50(path)
        if self.tickers:
            wanted = {safe_ticker(t): t for t in self.tickers}
            by_symbol = {safe_ticker(m.symbol): m for m in members}
            members = [
                by_symbol.get(key, Constituent(symbol=raw.strip().upper()))
                for key, raw in wanted.items()
            ]
        if self.limit is not None and self.limit >= 0:
            members = members[: self.limit]
        return members


# --- Records ---------------------------------------------------------------


@dataclass
class DocumentOutcome:
    """What happened to one catalogue entry in one run."""

    doc_id: str
    doc_type: str = ""
    label: str = ""
    reason: str = ""  # why Layer 1 selected it; "" when unchanged
    status: str = ""  # INDEXED | SKIPPED | FAILED | PENDING (dry run)
    stage: str = ""  # stage that failed, when status == FAILED
    chunks: int = 0
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickerRun:
    """One ticker's outcome in one run; the manifest's per-ticker entry."""

    symbol: str
    name: str = ""
    status: str = ""
    started_at: str = ""
    finished_at: str = ""
    catalog_entries: int = 0
    selected_documents: int = 0
    new_documents: int = 0
    unchanged_documents: int = 0
    failed_documents: int = 0
    chunks_indexed: int = 0
    points_total: int = 0
    collection: str = ""
    elapsed_seconds: float = 0.0
    errors: list[dict[str, str]] = field(default_factory=list)
    documents: list[DocumentOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["documents"] = [d.to_dict() for d in self.documents]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickerRun:
        docs = [
            DocumentOutcome(
                **{
                    k: v
                    for k, v in d.items()
                    if k in DocumentOutcome.__dataclass_fields__
                }
            )
            for d in (data.get("documents") or [])
            if isinstance(d, dict)
        ]
        known = set(cls.__dataclass_fields__) - {"documents"}
        return cls(**{k: v for k, v in data.items() if k in known}, documents=docs)


@dataclass
class RunManifest:
    """The cross-ticker record of a run, persisted after every ticker.

    Derived, regenerable state: it can always be rebuilt from the fifty
    ``state.json`` files plus a fresh catalogue fetch, so losing it costs run
    history and timings, never data.
    """

    run_id: str
    status: str = RUNNING
    pid: int = 0
    started_at: str = ""
    finished_at: str = ""
    notified_at: str = ""
    # False while the record is only the launcher's handoff note, True once the
    # process that actually runs the batch has taken it over. `pid` alone cannot
    # carry this: on Windows a venv's python.exe may be a launcher stub, so the
    # `Popen.pid` the launcher records is the stub's and never equals the batch
    # process's own `os.getpid()`.
    claimed: bool = False
    total_tickers: int = 0
    completed_tickers: int = 0
    embedding_model: str = ""
    collection_mode: str = ""
    qdrant_target: str = ""
    nifty50_as_of: str = ""
    log_dir: str = ""
    error: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    # The device, threads and batch size the run actually got, as opposed to
    # the ones it asked for -- the difference between a fifty-hour run and a
    # ten-hour one, and invisible afterwards unless it is written down here.
    hardware: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    elapsed_by_stage_seconds: dict[str, float] = field(default_factory=dict)
    tickers: dict[str, TickerRun] = field(default_factory=dict)
    # Whether this run wrote checkpoints, and the run it continued. Additive:
    # `--status` and the REST payload gain two keys and lose none.
    resumable: bool = False
    resumed_from: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def elapsed_seconds(self) -> float:
        try:
            start = _dt.datetime.fromisoformat(self.started_at)
            end = (
                _dt.datetime.fromisoformat(self.finished_at)
                if self.finished_at
                else _dt.datetime.now(_dt.UTC)
            )
            return max(0.0, (end - start).total_seconds())
        except ValueError:
            return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resumable": self.resumable,
            "resumed_from": self.resumed_from,
            "run_id": self.run_id,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notified_at": self.notified_at,
            "claimed": self.claimed,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_tickers": self.total_tickers,
            "completed_tickers": self.completed_tickers,
            "embedding_model": self.embedding_model,
            "collection_mode": self.collection_mode,
            "qdrant_target": self.qdrant_target,
            "nifty50_as_of": self.nifty50_as_of,
            "log_dir": self.log_dir,
            "error": self.error,
            "options": self.options,
            "hardware": self.hardware,
            "counters": self.counters,
            "elapsed_by_stage_seconds": self.elapsed_by_stage_seconds,
            "tickers": {k: v.to_dict() for k, v in self.tickers.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        tickers = {
            str(sym): TickerRun.from_dict(run)
            for sym, run in (data.get("tickers") or {}).items()
            if isinstance(run, dict)
        }
        return cls(
            run_id=str(data.get("run_id", "")),
            status=str(data.get("status", "")),
            pid=int(data.get("pid") or 0),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            notified_at=str(data.get("notified_at", "")),
            claimed=bool(data.get("claimed", False)),
            total_tickers=int(data.get("total_tickers") or 0),
            completed_tickers=int(data.get("completed_tickers") or 0),
            embedding_model=str(data.get("embedding_model", "")),
            collection_mode=str(data.get("collection_mode", "")),
            qdrant_target=str(data.get("qdrant_target", "")),
            nifty50_as_of=str(data.get("nifty50_as_of", "")),
            log_dir=str(data.get("log_dir", "")),
            error=str(data.get("error", "")),
            options=dict(data.get("options") or {}),
            hardware=dict(data.get("hardware") or {}),
            counters={str(k): int(v) for k, v in (data.get("counters") or {}).items()},
            elapsed_by_stage_seconds={
                str(k): float(v)
                for k, v in (data.get("elapsed_by_stage_seconds") or {}).items()
            },
            tickers=tickers,
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            resumable=bool(data.get("resumable", False)),
            resumed_from=data.get("resumed_from") or None,
        )

    def summary(self) -> dict[str, Any]:
        """The manifest without per-document detail, for notifications."""
        data = self.to_dict()
        data["tickers"] = {
            sym: {k: v for k, v in run.items() if k != "documents"}
            for sym, run in data["tickers"].items()
        }
        return data


# --- Manifest persistence ----------------------------------------------------


def manifest_dir(output_dir: Path | None = None) -> Path:
    """Returns ``<output>/_nifty50``."""
    return (output_dir or OUTPUT_DIR) / MANIFEST_DIR_NAME


def manifest_path(output_dir: Path | None = None) -> Path:
    """Returns the manifest's path for an output root."""
    return manifest_dir(output_dir) / MANIFEST_FILENAME


def run_manifest_path(run_id: str, output_dir: Path | None = None) -> Path:
    """Returns the archived per-run copy of a manifest."""
    return manifest_dir(output_dir) / "runs" / f"{run_id}.json"


def save_manifest(manifest: RunManifest, output_dir: Path | None = None) -> Path:
    """Atomically writes the live manifest and its per-run archive copy."""
    payload = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    for target in (
        manifest_path(output_dir),
        run_manifest_path(manifest.run_id, output_dir),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        try:
            temp.write_text(payload, encoding="utf-8")
            temp.replace(target)
        except Exception:
            with contextlib.suppress(OSError):
                temp.unlink()
            raise
    return manifest_path(output_dir)


def load_manifest(
    run_id: str | None = None, output_dir: Path | None = None
) -> RunManifest | None:
    """Reads the live manifest, or a run's archived copy; None when absent or corrupt."""
    path = (
        run_manifest_path(run_id, output_dir) if run_id else manifest_path(output_dir)
    )
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return RunManifest.from_dict(data)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read manifest %s: %s", path, exc)
        return None


def list_run_ids(output_dir: Path | None = None) -> list[str]:
    """Returns archived run ids, newest first."""
    folder = manifest_dir(output_dir) / "runs"
    if not folder.exists():
        return []
    return sorted((p.stem for p in folder.glob("*.json")), reverse=True)


# --- Single-flight -----------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running.

    ``os.kill(pid, 0)`` is the POSIX idiom; on Windows that call would
    *terminate* the process, so the handle-based check is used there.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() not in (
            psutil.STATUS_ZOMBIE,
            psutil.STATUS_DEAD,
        )
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - psutil raised on a vanished process
        return False

    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        query_limited = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(query_limited, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_stale(manifest: RunManifest, hours: int = STALE_RUN_HOURS) -> bool:
    """Whether a RUNNING manifest is old enough to be presumed abandoned."""
    try:
        started = _dt.datetime.fromisoformat(manifest.started_at)
    except ValueError:
        return True
    return _dt.datetime.now(_dt.UTC) - started > _dt.timedelta(hours=hours)


def active_run(output_dir: Path | None = None) -> RunManifest | None:
    """Returns the manifest of a run that is genuinely still executing, else None."""
    manifest = load_manifest(output_dir=output_dir)
    if manifest is None or manifest.status != RUNNING:
        return None
    if manifest.pid:
        # A recorded pid is authoritative: alive means live, dead means abandoned.
        return manifest if pid_alive(manifest.pid) else None
    # No pid recorded: trust the manifest until it is old enough to be a crash.
    return None if is_stale(manifest) else manifest


# --- Per-ticker pipeline -----------------------------------------------------


def resolve_workers(requested: int, hardware: HardwareProfile) -> int:
    """Returns the worker count this machine can actually sustain.

    Workers are threads sharing one process, so on a GPU every worker puts its
    own copy of the layout and table models on the same card, alongside the
    embedding model that is already resident. Four workers on a 4 GB card is
    an out-of-memory error some hours into a run that reports nothing until it
    happens, which is strictly worse than running one ticker at a time.

    Concurrency is also worth less on a GPU than on a CPU: the models are the
    bottleneck and they are already saturating the device, so a second ticker
    mostly waits.

    Args:
        requested: What the caller asked for.
        hardware: The resolved profile for this run.

    Returns:
        The worker count to use, in ``[1, MAX_WORKERS]``.
    """
    workers = min(max(1, requested), MAX_WORKERS)
    if workers == 1 or not hardware.device.startswith("cuda"):
        return workers
    if 0 < hardware.vram_gb < GPU_CONCURRENCY_VRAM_GB:
        logger.warning(
            "workers=%d requested, but %s has only %g GB of VRAM: every worker "
            "loads its own layout and table models onto the same card. "
            "Running one ticker at a time instead.",
            workers,
            hardware.gpu_name or hardware.device,
            hardware.vram_gb,
        )
        return 1
    return workers


def _qdrant_target(config: IndexerConfig) -> str:
    """Host of the configured Qdrant server, never its key."""
    if not config.qdrant_url:
        return "memory"
    parts = urlsplit(config.qdrant_url)
    return parts.netloc or config.qdrant_url


def _fail_ticker(
    run: TickerRun,
    runlog: RunLogger,
    started: float,
    status: str,
    message: str,
) -> TickerRun:
    """Closes a ticker that never got past its catalogue."""
    run.status = status
    if message:
        run.errors.append({"doc_id": "", "stage": "catalog", "error": message})
    run.elapsed_seconds = round(time.perf_counter() - started, 2)
    run.finished_at = utc_now()
    runlog.event(
        "ticker_completed",
        ticker=run.symbol,
        status=status,
        error=message,
        elapsed_s=run.elapsed_seconds,
    )
    return run


def _finish_dry_run(
    run: TickerRun,
    runlog: RunLogger,
    started: float,
    diff: stages.DiffResult,
) -> TickerRun:
    """Closes a ticker that only reported what it would have done."""
    for entry, reason in diff.pending:
        run.documents.append(
            DocumentOutcome(
                doc_id=entry.doc_id,
                doc_type=entry.doc_type,
                label=entry.label,
                reason=reason,
                status="PENDING",
            )
        )
    run.new_documents = len(diff.pending)
    run.status = TICKER_DRY_RUN
    runlog.add("tickers_dry_run")
    run.elapsed_seconds = round(time.perf_counter() - started, 2)
    run.finished_at = utc_now()
    runlog.event(
        "ticker_completed",
        ticker=run.symbol,
        status=run.status,
        new_documents=len(diff.pending),
        unchanged_documents=len(diff.unchanged),
        elapsed_s=run.elapsed_seconds,
    )
    return run


def _finish_unchanged(
    run: TickerRun,
    runlog: RunLogger,
    started: float,
    known_documents: list[str],
) -> TickerRun:
    """Closes a ticker whose catalogue check was skipped as still current.

    Reports the known filings as SKIPPED and leaves `catalog_entries` at zero,
    because no catalogue was read: the manifest says what the run did, and this
    run made no request. The `catalog_skipped` event carries the age and the
    doc ids the decision rested on.
    """
    run.unchanged_documents = len(known_documents)
    for doc_id in known_documents:
        run.documents.append(DocumentOutcome(doc_id=doc_id, status="SKIPPED"))
    run.status = TICKER_OK
    runlog.add("tickers_ok")
    runlog.add("tickers_catalogue_skipped")
    run.elapsed_seconds = round(time.perf_counter() - started, 2)
    run.finished_at = utc_now()
    runlog.event(
        "ticker_completed",
        ticker=run.symbol,
        status=run.status,
        new_documents=0,
        unchanged_documents=run.unchanged_documents,
        catalogue_skipped=True,
        elapsed_s=run.elapsed_seconds,
    )
    return run


def _record_failure(
    run: TickerRun,
    outcomes: dict[str, DocumentOutcome],
    doc_id: str,
    stage: str,
    error: str,
    seconds: float,
) -> None:
    """Records a stage failure against the one document it belongs to.

    A module-level function taking what it mutates, rather than a closure over
    the enclosing run: the same bookkeeping is reached from a sequential loop
    here and from concurrent graph tasks elsewhere, and shared enclosing state
    is how that becomes a race.
    """
    outcome = outcomes[doc_id]
    outcome.status = "FAILED"
    outcome.stage = stage
    outcome.error = error[:500]
    outcome.seconds = round(outcome.seconds + seconds, 2)
    run.errors.append({"doc_id": doc_id, "stage": stage, "error": error[:500]})


def _record_unchanged(run: TickerRun, diff: stages.DiffResult) -> None:
    """Records the documents Layer 1 found already current."""
    run.unchanged_documents = len(diff.unchanged)
    for entry in diff.unchanged:
        run.documents.append(
            DocumentOutcome(
                doc_id=entry.doc_id,
                doc_type=entry.doc_type,
                label=entry.label,
                status="SKIPPED",
                chunks=diff.chunk_count(entry.doc_id),
            )
        )


def _ingest_document(
    ctx: stages.StageContext,
    run: TickerRun,
    outcomes: dict[str, DocumentOutcome],
    extractor: Any,
    entry: Any,
    result: Any,
) -> bool:
    """Takes one downloaded filing through extraction and chunking.

    The three stages are consecutive because each consumes the last one's
    output, and a failure in any of them is recorded against this document
    alone -- the ticker's other filings are unaffected.

    Args:
        ctx: The ticker's stage context.
        run: The ticker's record, for its error list.
        outcomes: Per-document outcomes being built, keyed by doc id.
        extractor: The shared Docling extractor.
        entry: The catalogued filing.
        result: Its `DownloadResult`.

    Returns:
        True when a chunk file was written, so Layer 2's verdict for this
        document can be read as "content unchanged" rather than "skipped".
    """
    if not stages.note_download(ctx, result) or result.path is None:
        _record_failure(
            run, outcomes, entry.doc_id, "download", result.error, result.seconds
        )
        return False
    outcomes[entry.doc_id].seconds += result.seconds

    extracted = stages.extract_one(
        ctx, extractor, entry, result.path, source_sha256=result.sha256
    )
    if not extracted.ok or extracted.document is None:
        _record_failure(
            run, outcomes, entry.doc_id, "extract", extracted.error, extracted.seconds
        )
        return False
    outcomes[entry.doc_id].seconds += extracted.seconds

    chunked = stages.chunk_one(ctx, entry, extracted.document)
    if not chunked.ok:
        _record_failure(
            run, outcomes, entry.doc_id, "chunk", chunked.error, chunked.seconds
        )
        return False
    outcomes[entry.doc_id].seconds += chunked.seconds
    return True


def _fold_index_results(
    run: TickerRun,
    outcomes: dict[str, DocumentOutcome],
    indexed: stages.IndexStageResult,
    chunk_ready: set[str],
) -> None:
    """Folds Layer 2's verdicts into the outcomes this run is building.

    Chunk files for documents this run did not select are indexed too -- that
    is how a document that failed to embed last time recovers -- but they have
    no outcome to fold into, so they are skipped here.
    """
    for doc in indexed.documents:
        outcome = outcomes.get(doc.doc_id)
        if outcome is None or outcome.status == "FAILED":
            continue
        outcome.status = doc.status
        outcome.chunks = doc.chunks_indexed
        outcome.seconds = round(outcome.seconds + doc.seconds, 2)
        if doc.status == "FAILED":
            _record_failure(
                run, outcomes, doc.doc_id, "index", doc.error or "index failed", 0.0
            )
        elif doc.status == "SKIPPED" and doc.doc_id in chunk_ready:
            # Re-extracted to an identical fingerprint: content unchanged.
            outcome.status = "INDEXED"


def _finalise(
    run: TickerRun,
    runlog: RunLogger,
    started: float,
    outcomes: dict[str, DocumentOutcome],
    indexed_any: bool,
) -> TickerRun:
    """Derives the ticker's terminal status from its documents' outcomes."""
    run.new_documents = sum(1 for o in outcomes.values() if o.status == "INDEXED")
    run.failed_documents = sum(1 for o in outcomes.values() if o.status == "FAILED")
    if run.failed_documents == 0:
        run.status = TICKER_OK
        runlog.add("tickers_ok")
    elif run.new_documents > 0 or run.unchanged_documents > 0:
        run.status = TICKER_PARTIAL
        runlog.add("tickers_partial")
    else:
        run.status = TICKER_FAILED
        runlog.add("tickers_failed")
    if indexed_any:
        runlog.add("collections_touched")

    run.elapsed_seconds = round(time.perf_counter() - started, 2)
    run.finished_at = utc_now()
    runlog.event(
        "ticker_completed",
        ticker=run.symbol,
        status=run.status,
        new_documents=run.new_documents,
        unchanged_documents=run.unchanged_documents,
        failed_documents=run.failed_documents,
        chunks_indexed=run.chunks_indexed,
        points_total=run.points_total,
        elapsed_s=run.elapsed_seconds,
        errors=run.errors,
    )
    return run


def process_ticker(
    member: Constituent,
    options: BatchOptions,
    indexer: QdrantVectorIndexer,
    runlog: RunLogger,
    position: int = 0,
    total: int = 0,
) -> TickerRun:
    """Runs catalogue -> diff -> download -> extract -> chunk -> index for one ticker.

    Composes the stages in `ingestion.stages` sequentially and folds their
    results into one `TickerRun`. The stages do the work and own their
    logging; everything here is orchestration -- what a failed stage means for
    the ticker, and what the ticker's terminal status is.

    Never raises: every failure is recorded against the document or ticker it
    belongs to so the batch carries on with the rest.

    Args:
        member: The constituent to process.
        options: The run's options.
        indexer: The vector store, shared across tickers.
        runlog: The run's logger.
        position: 1-based index of this ticker in the run, for the log.
        total: Tickers in the run, for the log.

    Returns:
        The finished record for this ticker.
    """
    ctx = stages.StageContext.build(member, options, indexer, runlog)
    started = time.perf_counter()
    run = TickerRun(
        symbol=ctx.symbol,
        name=member.name,
        started_at=utc_now(),
        collection=ctx.collection,
    )
    runlog.event(
        "ticker_started",
        ticker=ctx.symbol,
        index=position,
        total=total,
        symbol=member.symbol,
        company_name=member.name,
        forced=ctx.forced,
    )

    # Layer 1 input: the catalogue.
    catalogue = stages.fetch_catalogue(ctx)
    if not catalogue.ok or catalogue.catalog is None:
        if catalogue.no_documents:
            runlog.add("tickers_no_documents")
            return _fail_ticker(run, runlog, started, TICKER_NO_DOCUMENTS, "")
        runlog.add("tickers_failed")
        return _fail_ticker(run, runlog, started, TICKER_FAILED, catalogue.error)

    if catalogue.skipped:
        return _finish_unchanged(run, runlog, started, catalogue.known_documents)

    run.catalog_entries = catalogue.entries_total
    if catalogue.company_name and not run.name:
        run.name = catalogue.company_name

    # Layer 1: what this run must actually process.
    diff = stages.compute_diff(ctx, catalogue.catalog)
    run.selected_documents = len(diff.selected)
    _record_unchanged(run, diff)
    if options.dry_run:
        return _finish_dry_run(run, runlog, started, diff)

    outcomes: dict[str, DocumentOutcome] = {
        entry.doc_id: DocumentOutcome(
            doc_id=entry.doc_id,
            doc_type=entry.doc_type,
            label=entry.label,
            reason=reason,
        )
        for entry, reason in diff.pending
    }
    by_id = {entry.doc_id: entry for entry, _ in diff.pending}
    chunk_ready: set[str] = set()

    if diff.pending:
        downloads = stages.download_pending(ctx, diff.pending)
        extractor = stages.build_extractor(ctx)
        for result in downloads.results:
            entry = by_id[result.doc_id]
            if _ingest_document(ctx, run, outcomes, extractor, entry, result):
                chunk_ready.add(entry.doc_id)

    # Layer 2: embed every chunk file on disk, not only this run's.
    indexed = stages.index_chunk_files(ctx, pending_ids=set(by_id))
    run.chunks_indexed = indexed.chunks_indexed
    _fold_index_results(run, outcomes, indexed, chunk_ready)

    for doc_id, outcome in outcomes.items():
        if not outcome.status:
            # Downloaded/extracted but its chunk file never appeared.
            _record_failure(
                run,
                outcomes,
                doc_id,
                outcome.stage or "chunk",
                "no chunk file produced",
                0.0,
            )
        run.documents.append(outcome)

    run.points_total = stages.verify_and_mirror(ctx, indexed.indexed_any).points_total
    return _finalise(run, runlog, started, outcomes, indexed.indexed_any)


# --- Resuming ----------------------------------------------------------------

# Settings a resume may not disagree with the original run about, because each
# one changes the text that would be produced or where it would be stored, so
# continuing under a different value would leave one corpus built two ways.
# `fast_tables` is the sharpest: it moves the recorded extractor version, so
# flipping it re-extracts every filing already indexed the other way -- across
# all fifty tickers.
RESUME_LOCKED_KEYS: tuple[str, ...] = (
    "fast_tables",
    "page_filter",
    "annual_reports",
    "transcripts",
    "presentations",
    "concall_years",
    "chunk_size",
    "chunk_overlap",
)

# Settings a resume may freely differ on: none of them changes the produced
# text or vectors, and picking a stalled run back up on a bigger machine is a
# thing an operator legitimately wants to do.
RESUME_FREE_KEYS: tuple[str, ...] = (
    "device",
    "num_threads",
    "embed_batch_size",
    "workers",
    "delay_seconds",
    "verbose",
    "notify",
    "notify_per_ticker",
    "webhook_url",
    "catalog_ttl_hours",
    "allow_ephemeral",
    "no_checkpoint",
    "output_dir",
    "log_dir",
)


class ResumeMismatch(BatchError):
    """Raised when a resume would continue a run under different settings."""


def merge_resume_options(
    stored: dict[str, Any], requested: BatchOptions
) -> BatchOptions:
    """Returns the options a resume should run with, or refuses.

    A resume takes the original run's options and overlays only the keys that
    cannot change what the run produces. Pure, so the refusal rules are
    testable without a run.

    Args:
        stored: The `options` dict the resumed run's manifest recorded.
        requested: The options this invocation was given.

    Returns:
        The merged options, carrying the original scope and the new compute
        settings.

    Raises:
        ResumeMismatch: If a locked setting differs. The message names the key,
            because the fix is to start a fresh run rather than to guess.
    """
    base = BatchOptions.from_dict(stored)
    differing = [
        key
        for key in RESUME_LOCKED_KEYS
        if getattr(base, key, None) != getattr(requested, key, None)
    ]
    if differing:
        detail = ", ".join(
            f"{key}: run has {getattr(base, key, None)!r}, "
            f"you asked for {getattr(requested, key, None)!r}"
            for key in differing
        )
        raise ResumeMismatch(
            f"cannot resume under different settings ({detail}). "
            "Start a new run instead."
        )
    for key in RESUME_FREE_KEYS:
        if hasattr(requested, key):
            setattr(base, key, getattr(requested, key))
    base.resume = requested.resume
    base.run_id = requested.resume
    return base


# --- Driving the tickers -----------------------------------------------------


def _drive(
    members: list[Constituent],
    options: BatchOptions,
    indexer: QdrantVectorIndexer,
    runlog: RunLogger,
    workers: int,
    record: Callable[[TickerRun], None],
    resume_from: str | None = None,
) -> None:
    """Runs every ticker in scope, checkpointing progress unless told not to.

    Two ways to get through the constituent list, and the difference is only
    durability: the graph writes the dispatch queue and every finished ticker
    to disk after each batch, so `--resume` can pick the run up where it
    stopped. `--no-checkpoint` reproduces the plain loop exactly, which is the
    escape hatch if the saver ever misbehaves in production.

    Args:
        members: Constituents in scope, in dispatch order.
        options: The run's options.
        indexer: The vector store, shared across tickers.
        runlog: The run's logger.
        workers: Tickers to run concurrently, already capped.
        record: Called with each finished ticker.
        resume_from: A run id whose checkpoint this run continues. None starts
            the thread fresh.
    """
    if options.no_checkpoint:
        _drive_sequentially(members, options, indexer, runlog, workers, record)
        return

    from ingestion.graph import GraphResources, JsonFileSaver, build_batch_graph

    order = [safe_ticker(m.symbol) for m in members]
    resources = GraphResources(
        options=options,
        indexer=indexer,
        runlog=runlog,
        workers=workers,
        members={safe_ticker(m.symbol): m for m in members},
        order=order,
        record=record,
    )
    saver = JsonFileSaver(output_dir=options.output_root)
    graph = build_batch_graph(resources, checkpointer=saver)
    thread_id = resume_from or options.run_id or new_run_id()
    config = {"configurable": {"thread_id": thread_id}}

    # None as the input is LangGraph's resume-from-checkpoint form; a fresh
    # run seeds the queue instead. `durability="sync"` because the default,
    # "async", can lose the last superstep to exactly the hard kill this is
    # built for -- one small file write per batch against Docling passes
    # measured in minutes is not a trade worth making.
    resuming = resume_from is not None and saver.get_tuple(config) is not None
    if resume_from is not None and not resuming:
        logger.warning(
            "no checkpoint found for run %s; starting it from the beginning.",
            resume_from,
        )
    if not resuming and saver.get_tuple(config) is not None:
        # A fresh run must never adopt another run's queue. It can happen:
        # `new_run_id` has one-second resolution, so two runs started in the
        # same second share an id, and the thread id is the run id. Without
        # this the second run would resume the first one's dispatch queue and
        # report its results as its own.
        logger.warning(
            "run %s already has checkpoints; discarding them, because this is a "
            "new run rather than a resume.",
            thread_id,
        )
        saver.delete_thread(thread_id)

    payload = None if resuming else {"queue": order, "in_flight": []}
    graph.invoke(payload, config, durability="sync")


def _drive_sequentially(
    members: list[Constituent],
    options: BatchOptions,
    indexer: QdrantVectorIndexer,
    runlog: RunLogger,
    workers: int,
    record: Callable[[TickerRun], None],
) -> None:
    """Runs the tickers in a plain loop, with no checkpointing.

    Kept as the reference behaviour `--no-checkpoint` selects, and as what the
    verification suite compares the graph against.
    """

    def work(item: tuple[int, Constituent]) -> None:
        position, member = item
        if position > 1 and options.delay_seconds > 0:
            time.sleep(options.delay_seconds)
        record(process_ticker(member, options, indexer, runlog, position, len(members)))

    items = list(enumerate(members, start=1))
    if workers == 1:
        for item in items:
            work(item)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ticker") as pool:
        list(pool.map(work, items))


# --- The batch ---------------------------------------------------------------


def run_batch(
    options: BatchOptions,
    on_ticker: Callable[[TickerRun, RunManifest], None] | None = None,
) -> RunManifest:
    """Embeds every ticker in scope and returns the finished manifest.

    Args:
        options: What to run.
        on_ticker: Called after each ticker with its result and the manifest so
            far -- the CLI prints its per-ticker table from here.

    Raises:
        BatchAlreadyRunning: If another run is live in the manifest.
        BatchError: If the constituent list cannot be loaded.
    """
    from core.hardware import configure as configure_hardware
    from ingestion.indexer import IndexerConfig, QdrantVectorIndexer
    from ingestion.notify import notify_completion, notify_ticker

    output_dir = options.output_root
    live = active_run(output_dir)
    # An unclaimed manifest is `launch_detached`'s handoff note for the child it
    # just spawned -- this process -- not a rival run, so claiming it is right.
    # It cannot mask a genuine double-launch: `launch_detached` runs this same
    # `active_run` check before spawning anything, and a batch that is really
    # under way has already set `claimed`.
    if live is not None and live.claimed and live.pid != os.getpid():
        raise BatchAlreadyRunning(
            f"run {live.run_id} is still RUNNING (pid {live.pid}); "
            f"started {live.started_at}"
        )

    try:
        members = options.resolve_constituents()
    except Exception as exc:
        raise BatchError(f"could not load constituents: {exc}") from exc

    # Resuming reuses the run id, so the manifest, the log directory and the
    # checkpoint thread all stay one logical run. `run.log` and `events.jsonl`
    # are appended to; the second `run_started` carries `resumed_from` so the
    # event stream is unambiguous when it is read back with `jq`.
    resume_from = (options.resume or "").strip() or None
    run_id = resume_from or options.run_id or new_run_id()
    options.run_id = run_id
    runlog = setup_run_logging(
        JOB_NAME, run_id=run_id, root=options.log_root, verbose=options.verbose
    )

    # Before any model is built: the libraries underneath read their thread and
    # device settings once, when they first construct something.
    hardware = configure_hardware(
        device=options.device,
        num_threads=options.num_threads,
        embed_batch_size=options.embed_batch_size,
    )
    logger.info("Hardware: %s", hardware.summary())
    workers = resolve_workers(options.workers, hardware)

    config = IndexerConfig.from_env(
        output_dir=output_dir,
        device=hardware.device,
        batch_size=hardware.embed_batch_size,
        fp16=hardware.embed_fp16,
    )
    indexer = QdrantVectorIndexer(config=config)

    # Before anything is downloaded, and before a manifest exists to record a
    # run that should not start. An in-memory store loses every vector when the
    # process exits, while `state.json` still records them as INDEXED -- so the
    # next run skips them and the corpus is quietly empty. That is the one
    # failure mode where resuming is wrong rather than slow, so it refuses
    # rather than warns. Raised here, not in a stage, so the `BatchError`
    # contract the CLI and the API already handle is unchanged.
    if not indexer.is_persistent and not options.dry_run:
        if not options.allow_ephemeral:
            raise BatchError(
                "Qdrant is in-memory for this run: vectors would not survive the "
                "process, but state.json would record them as INDEXED, so the next "
                "run would skip them. Set QDRANT_API_URL (and QDRANT_API_KEY) to a "
                "Qdrant server, or pass --allow-ephemeral to accept that."
            )
        logger.error(
            "Qdrant is in-memory for this run: vectors will not survive the process. "
            "Proceeding because --allow-ephemeral was given."
        )

    manifest = RunManifest(
        run_id=run_id,
        status=RUNNING,
        pid=os.getpid(),
        claimed=True,
        started_at=utc_now(),
        total_tickers=len(members),
        embedding_model=config.model_alias,
        collection_mode=(
            f"per-ticker ({config.collection_prefix}_<ticker>)"
            if config.collection_per_ticker
            else f"shared ({config.collection_name})"
        ),
        qdrant_target=_qdrant_target(config),
        nifty50_as_of=verified_as_of(members),
        log_dir=str(runlog.run_dir),
        options=options.to_dict(),
        hardware=hardware.to_dict(),
        resumable=not options.no_checkpoint,
        resumed_from=resume_from,
    )
    if resume_from:
        # One run id is one logical run, so its manifest accumulates. Without
        # this the resumed run would report only the tickers this invocation
        # touched and appear to have lost the ones the first attempt finished
        # -- which is what `--status` and the REST payload would then show.
        previous = load_manifest(run_id=resume_from, output_dir=output_dir)
        if previous is not None:
            manifest.tickers.update(previous.tickers)
            manifest.completed_tickers = len(manifest.tickers)
    save_manifest(manifest, output_dir)
    runlog.write_latest(RUNNING, manifest=str(manifest_path(output_dir)))
    runlog.event(
        "run_started",
        tickers_in_scope=[m.symbol for m in members],
        total=len(members),
        mode="dry-run" if options.dry_run else "embed",
        resumed_from=resume_from,
        checkpointed=not options.no_checkpoint,
        force_all=options.force_all,
        force=options.force,
        annual_reports=options.annual_reports,
        transcripts=options.transcripts,
        presentations=options.presentations,
        concall_years=options.concall_years,
        workers=workers,
        workers_requested=options.workers,
        fast_tables=options.fast_tables,
        page_filter=options.page_filter,
        hardware=hardware.to_dict(),
        embedding_model=config.model_name,
        qdrant_target=manifest.qdrant_target,
        qdrant_persistent=indexer.is_persistent,
        collection_mode=manifest.collection_mode,
        nifty50_as_of=manifest.nifty50_as_of,
        output_dir=str(output_dir),
        log_dir=str(runlog.run_dir),
        pid=os.getpid(),
        python=sys.version.split()[0],
        platform=sys.platform,
    )
    lock = threading.Lock()
    started = time.perf_counter()

    notify_progress = (
        options.notify and options.notify_per_ticker and not options.dry_run
    )

    def finish_ticker(run: TickerRun) -> None:
        with lock:
            manifest.tickers[run.symbol] = run
            manifest.completed_tickers = len(manifest.tickers)
            snap = runlog.snapshot()
            manifest.counters = snap["counters"]
            manifest.elapsed_by_stage_seconds = snap["elapsed_by_stage_seconds"]
            save_manifest(manifest, output_dir)
            position = manifest.completed_tickers
        if on_ticker is not None:
            with contextlib.suppress(Exception):
                on_ticker(run, manifest)
        # Announced here rather than from `on_ticker` so a detached run and an
        # API-launched one get the same feed: the CLI's callback is the console
        # table, and a background child passes none. Outside the lock because
        # this is a network round trip -- a slow webhook must not hold up the
        # next ticker, and it has no business serialising the manifest write.
        if notify_progress:
            result = notify_ticker(
                run.to_dict(),
                position=position,
                total=manifest.total_tickers,
                webhook_url=options.webhook_url,
            )
            if result is not None:
                runlog.event(
                    "ticker_notify_sent" if result.ok else "ticker_notify_failed",
                    ticker=run.symbol,
                    channel=result.channel,
                    detail=result.detail,
                    status=run.status,
                )

    try:
        _drive(
            members=members,
            options=options,
            indexer=indexer,
            runlog=runlog,
            workers=workers,
            record=finish_ticker,
            resume_from=resume_from,
        )
        statuses = {run.status for run in manifest.tickers.values()}
        bad = statuses & {TICKER_FAILED, TICKER_PARTIAL}
        manifest.status = COMPLETED_WITH_ERRORS if bad else COMPLETED
    except BaseException as exc:  # noqa: BLE001 - record the crash, then re-raise
        manifest.status = FAILED
        manifest.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Batch run %s aborted", run_id)
        raise
    finally:
        manifest.finished_at = utc_now()
        snap = runlog.snapshot()
        manifest.counters = snap["counters"]
        manifest.elapsed_by_stage_seconds = snap["elapsed_by_stage_seconds"]
        save_manifest(manifest, output_dir)
        runlog.event(
            "run_completed",
            status=manifest.status,
            total_tickers=manifest.total_tickers,
            completed_tickers=manifest.completed_tickers,
            elapsed_s=round(time.perf_counter() - started, 1),
            error=manifest.error,
            **snap,
        )
        summary = manifest.summary()
        summary["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        runlog.write_summary(summary)

        # Notify exactly once, after every ticker's Qdrant upserts have been
        # acknowledged (each upsert waits for the server) and counted.
        if not manifest.notified_at:
            results = notify_completion(
                summary,
                webhook_url=options.webhook_url,
                enabled=options.notify and not options.dry_run,
            )
            manifest.notified_at = utc_now()
            for res in results:
                runlog.event(
                    "notify_sent" if res.ok else "notify_failed",
                    channel=res.channel,
                    detail=res.detail,
                )
            save_manifest(manifest, output_dir)
        runlog.write_latest(manifest.status, manifest=str(manifest_path(output_dir)))
        runlog.close()

    return manifest


# --- Detached launch ---------------------------------------------------------


def write_options_file(options: BatchOptions, run_id: str) -> Path:
    """Persists options for a detached child under the run's log directory."""
    from ingestion.runlog import LOG_ROOT

    root = options.log_root or LOG_ROOT
    folder = root / JOB_NAME / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "options.json"
    path.write_text(
        json.dumps(options.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def launch_background(options: BatchOptions) -> tuple[str, Path, int]:
    """Starts `run_batch` in a detached process and returns immediately.

    The child is ``python -m scripts.embed_nifty50 --options-file <path>``; it
    survives this process exiting, and its raw stdout/stderr land in the run's
    log directory so a hard crash still leaves a traceback on disk.

    Returns:
        ``(run_id, stdout_log_path, pid)``.

    Raises:
        BatchAlreadyRunning: If a live run is recorded in the manifest.
    """
    output_dir = options.output_root
    live = active_run(output_dir)
    if live is not None:
        raise BatchAlreadyRunning(
            f"run {live.run_id} is still RUNNING (pid {live.pid}); started {live.started_at}"
        )

    run_id = options.run_id or new_run_id()
    options.run_id = run_id
    options_path = write_options_file(options, run_id)
    stdout_path = options_path.with_name("stdout.log")

    command = [
        sys.executable,
        "-m",
        "scripts.embed_nifty50",
        "--options-file",
        str(options_path),
    ]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with stdout_path.open("ab") as out:
        proc = subprocess.Popen(
            command, stdout=out, stderr=subprocess.STDOUT, env=env, **popen_kwargs
        )

    # Record the launch so a status poll issued before the child writes its
    # own manifest still sees a RUNNING run with a live pid.
    manifest = RunManifest(
        run_id=run_id,
        status=RUNNING,
        pid=proc.pid,
        started_at=utc_now(),
        log_dir=str(options_path.parent),
        options=options.to_dict(),
    )
    save_manifest(manifest, output_dir)
    logger.info(
        "Launched background run %s (pid %d); log: %s", run_id, proc.pid, stdout_path
    )
    return run_id, stdout_path, proc.pid


# --- Status ------------------------------------------------------------------


def status_report(
    run_id: str | None = None, output_dir: Path | None = None
) -> dict[str, Any]:
    """Returns the manifest as JSON-ready data plus derived progress; no network.

    A RUNNING manifest whose process has died is reported as ``ABANDONED`` so
    a poller is not left waiting on it.
    """
    manifest = load_manifest(run_id=run_id, output_dir=output_dir)
    if manifest is None:
        return {
            "status": "NO_RUNS" if not run_id else "UNKNOWN_RUN",
            "run_id": run_id,
            "runs": list_run_ids(output_dir),
        }
    data = manifest.to_dict()
    if manifest.status == RUNNING:
        alive = bool(manifest.pid) and pid_alive(manifest.pid)
        if not alive and (manifest.pid or is_stale(manifest)):
            data["status"] = "ABANDONED"
        data["pid_alive"] = alive
    total = max(manifest.total_tickers, 1)
    data["progress"] = round(manifest.completed_tickers / total, 3)
    data["runs"] = list_run_ids(output_dir)
    return data


def admin_token_ok(presented: str | None) -> bool:
    """Whether a presented bearer token matches ``NIFTY50_ADMIN_TOKEN``.

    An unset token means the trigger is disabled, never open.
    """
    import hmac

    expected = os.getenv(ADMIN_TOKEN_ENV, "").strip()
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected, presented.strip())
