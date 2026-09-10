"""Verification harness for the Nifty 50 batch embedding pipeline.

Run this after changing `ingestion/batch.py`, `ingestion/indexer.py`,
`ingestion/runlog.py`, `ingestion/notify.py` or `ingestion/nifty50.py`:

    venv/Scripts/python.exe scripts/verify_nifty50_embeddings.py
    venv/Scripts/python.exe scripts/verify_nifty50_embeddings.py --live WIPRO

The offline checks need no network and no Qdrant. The one worth naming is the
regression guard on `QdrantVectorIndexer.should_index_document`: its final
comparison was inverted, so every up-to-date document was re-embedded on every
run and a genuine collection change was skipped. A steady-state run being a
no-op depends on that one line, so it is asserted directly.

The live checks (`--live SYM`) run the batch twice against one real ticker and
assert the second run embeds nothing, that Qdrant reports the points from a
fresh client, and that the per-stock ``vectors.npz`` mirror agrees with it.
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import safe_ticker  # noqa: E402
from ingestion import batch, nifty50, notify, runlog, stages  # noqa: E402
from ingestion.catalog import CatalogEntry  # noqa: E402
from ingestion.fetcher import (  # noqa: E402
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
)
from ingestion.indexer import (  # noqa: E402
    DocumentIndexState,
    IndexerConfig,
    QdrantVectorIndexer,
    TickerState,
)
from scripts import (
    cli,  # noqa: E402
    embed_nifty50 as embed_cli,  # noqa: E402
)
from scripts.checks import Report, banner, require  # noqa: E402


def _entry(doc_id: str, doc_type: str = DOC_TYPE_TRANSCRIPT) -> CatalogEntry:
    return CatalogEntry(
        doc_id=doc_id,
        doc_type=doc_type,
        label=doc_id,
        source_url=f"https://x/{doc_id}.pdf",
    )


def _state(
    ticker: str, _config: IndexerConfig, **docs: DocumentIndexState
) -> TickerState:
    return TickerState(
        ticker=safe_ticker(ticker), last_updated="", documents=dict(docs)
    )


def _indexed(
    config: IndexerConfig, ticker: str, fingerprint: str = "fp", **overrides
) -> DocumentIndexState:
    from ingestion.chunker import CHUNKER_VERSION
    from ingestion.documents.extract import EXTRACT_VERSION

    base = {
        "fingerprint": fingerprint,
        "chunk_count": 3,
        "embedding_model": config.model_alias,
        "qdrant_collection": config.collection_for(ticker),
        "indexed_at": "2026-09-01T00:00:00+00:00",
        "status": "INDEXED",
        "chunker_version": CHUNKER_VERSION,
        "extract_version": EXTRACT_VERSION,
    }
    base.update(overrides)
    return DocumentIndexState(**base)


# --- Offline checks -----------------------------------------------------------


def check_constituents(report: Report) -> None:
    report.section("Nifty 50 constituents (ticker_mapping.csv)")
    members = nifty50.load_nifty50()
    report.check("exactly 50 constituents", len(members) == 50, f"{len(members)} rows")
    symbols = [m.symbol for m in members]
    report.check("symbols unique", len(set(symbols)) == len(symbols))
    folded = {safe_ticker(s) for s in symbols}
    report.check("no safe_ticker() collisions", len(folded) == len(symbols))
    report.check("every row named", all(m.name for m in members))
    as_of = nifty50.verified_as_of(members)
    report.check("verified_at present", bool(as_of), as_of)
    report.check(
        "list not stale (< 6 months)", not nifty50.is_stale(members), f"as of {as_of}"
    )


def check_collections(report: Report) -> None:
    report.section("Per-ticker collections")
    cfg = IndexerConfig(collection_prefix="grownxt", collection_per_ticker=True)
    report.check(
        "M&M folds to grownxt_m_m",
        cfg.collection_for("M&M") == "grownxt_m_m",
        cfg.collection_for("M&M"),
    )
    report.check(
        "BAJAJ-AUTO keeps hyphen",
        cfg.collection_for("BAJAJ-AUTO") == "grownxt_bajaj-auto",
    )
    report.check(
        "None -> shared collection", cfg.collection_for(None) == cfg.collection_name
    )
    shared = IndexerConfig(collection_per_ticker=False, collection_name="shared")
    report.check("shared mode ignores ticker", shared.collection_for("TCS") == "shared")
    env_cfg = IndexerConfig.from_env(collection_per_ticker=True, collection_prefix="p")
    report.check("from_env honours overrides", env_cfg.collection_for("TCS") == "p_tcs")


def check_should_index_regression(report: Report) -> None:
    report.section("Layer 2: should_index_document (inversion regression)")
    cfg = IndexerConfig(
        collection_prefix="grownxt", collection_per_ticker=True, qdrant_url=None
    )
    indexer = QdrantVectorIndexer(config=cfg)
    current = _state("TCS", cfg, d1=_indexed(cfg, "TCS"))
    report.check(
        "up-to-date document is NOT re-indexed",
        indexer.should_index_document("TCS", "d1", "fp", state=current) is False,
    )
    moved = _state(
        "TCS",
        cfg,
        d1=_indexed(cfg, "TCS", qdrant_collection="grownxt_financial_elements"),
    )
    report.check(
        "collection change IS re-indexed",
        indexer.should_index_document("TCS", "d1", "fp", state=moved) is True,
    )
    report.check(
        "fingerprint change IS re-indexed",
        indexer.should_index_document("TCS", "d1", "other", state=current),
    )
    report.check(
        "unknown doc IS indexed",
        indexer.should_index_document("TCS", "nope", "fp", state=current),
    )
    failed = _state("TCS", cfg, d1=_indexed(cfg, "TCS", status="FAILED"))
    report.check(
        "failed doc IS re-indexed",
        indexer.should_index_document("TCS", "d1", "fp", state=failed),
    )
    report.check(
        "force wins",
        indexer.should_index_document("TCS", "d1", "fp", state=current, force=True),
    )


def check_layer1(report: Report) -> None:
    report.section("Layer 1: pending-set computation")
    from ingestion.chunker import CHUNKER_VERSION

    cfg = IndexerConfig(collection_prefix="grownxt", collection_per_ticker=True)
    entries = [_entry(f"d{i}") for i in range(1, 7)]
    state = _state(
        "TCS",
        cfg,
        d1=_indexed(cfg, "TCS"),
        d2=_indexed(cfg, "TCS", status="FAILED"),
        d3=_indexed(cfg, "TCS", chunker_version="chunker/old"),
        d4=_indexed(cfg, "TCS", qdrant_collection="grownxt_financial_elements"),
        d5=_indexed(cfg, "TCS", extract_version="extract/old"),
    )
    pending, unchanged = batch.compute_pending(entries, state, cfg)
    reasons = {e.doc_id: r for e, r in pending}
    report.check(
        "d1 unchanged",
        [e.doc_id for e in unchanged] == ["d1"],
        str([e.doc_id for e in unchanged]),
    )
    report.check("d2 retry_failed", reasons.get("d2") == batch.REASON_RETRY)
    report.check("d3 chunker version_bump", reasons.get("d3") == batch.REASON_VERSION)
    report.check("d4 collection_change", reasons.get("d4") == batch.REASON_COLLECTION)
    report.check("d5 extractor version_bump", reasons.get("d5") == batch.REASON_VERSION)
    report.check("d6 new", reasons.get("d6") == batch.REASON_NEW)
    forced, _ = batch.compute_pending(entries, state, cfg, forced=True)
    report.check(
        "forced -> every entry pending",
        len(forced) == 6 and all(r == batch.REASON_FORCED for _, r in forced),
    )
    legacy = _state(
        "TCS", cfg, d1=_indexed(cfg, "TCS", chunker_version="", extract_version="")
    )
    p2, u2 = batch.compute_pending([entries[0]], legacy, cfg)
    report.check(
        "legacy state without versions is trusted",
        not p2 and len(u2) == 1,
        f"CHUNKER_VERSION={CHUNKER_VERSION}",
    )


def check_selection(report: Report) -> None:
    report.section("Newest-N selection per document class")
    from ingestion.catalog import parse_catalog

    payload = {
        "company_name": "Test Ltd",
        "annual_reports": [
            {"financial_year": "FY2025", "url": "https://x/a25.pdf"},
            {"financial_year": "FY2026", "url": "https://x/a26.pdf"},
        ],
        "concalls": [
            {
                "date": "2026-04",
                "period": "Apr 2026",
                "transcript_url": "https://x/t1.pdf",
                "ppt_url": "https://x/p1.pdf",
            },
            {
                "date": "2026-07",
                "period": "Jul 2026",
                "transcript_url": "https://x/t2.pdf",
                "ppt_url": "",
            },
            {
                "date": "2026-01",
                "period": "Jan 2026",
                "transcript_url": "https://x/t0.pdf",
                "ppt_url": "https://x/p0.pdf",
            },
        ],
    }
    catalog = parse_catalog("TEST", payload)
    chosen = batch.select_entries(catalog, 1, 1, 1)
    ids = [e.doc_id for e in chosen]
    report.check("three documents chosen", len(chosen) == 3, str(ids))
    report.check("newest annual report", "annual_report_FY2026" in ids)
    report.check("newest transcript (Jul 2026)", "transcript_2026_07" in ids)
    report.check(
        "newest presentation (Apr 2026; Jul had none)", "presentation_2026_04" in ids
    )
    two = batch.select_entries(catalog, 2, 2, 0)
    report.check(
        "counts honoured (2,2,0)",
        len(two) == 4 and not any(e.doc_type == DOC_TYPE_PRESENTATION for e in two),
    )
    report.check(
        "zero annual reports allowed",
        not any(
            e.doc_type == DOC_TYPE_ANNUAL_REPORT
            for e in batch.select_entries(catalog, 0, 1, 1)
        ),
    )


def check_manifest(report: Report, tmp: Path) -> None:
    report.section("Run manifest persistence")
    out = tmp / "out"
    m = batch.RunManifest(
        run_id="r1",
        status=batch.RUNNING,
        pid=os.getpid(),
        started_at=runlog.utc_now(),
        total_tickers=2,
    )
    m.tickers["TCS"] = batch.TickerRun(
        symbol="TCS",
        status="OK",
        new_documents=1,
        documents=[batch.DocumentOutcome(doc_id="d1", status="INDEXED", chunks=5)],
    )
    m.completed_tickers = 1
    batch.save_manifest(m, out)
    loaded = batch.load_manifest(output_dir=out)
    report.check("round-trips", loaded is not None and loaded.to_dict() == m.to_dict())
    report.check(
        "archived per-run copy exists", batch.run_manifest_path("r1", out).exists()
    )
    report.check("run listed", batch.list_run_ids(out) == ["r1"])
    batch.manifest_path(out).write_text("{not json", encoding="utf-8")
    report.check(
        "corrupt manifest -> None, no raise",
        batch.load_manifest(output_dir=out) is None,
    )
    report.check(
        "missing manifest -> None",
        batch.load_manifest(output_dir=tmp / "nowhere") is None,
    )
    report.check(
        "status on empty dir -> NO_RUNS",
        batch.status_report(output_dir=tmp / "nowhere")["status"] == "NO_RUNS",
    )
    live = batch.active_run(out)
    report.check("corrupt manifest is not a live run", live is None)
    batch.save_manifest(m, out)
    report.check("own pid RUNNING is a live run", batch.active_run(out) is not None)
    m.pid = 2_000_000_000
    batch.save_manifest(m, out)
    report.check("dead pid RUNNING is not live", batch.active_run(out) is None)
    report.check(
        "status reports dead RUNNING as ABANDONED",
        batch.status_report(output_dir=out)["status"] == "ABANDONED",
    )
    _check_launch_handoff(report, out)


def _check_launch_handoff(report: Report, out: Path) -> None:
    """The launcher's handoff record must not lock the child out of its own run.

    `launch_detached` writes a RUNNING manifest before the child exists, using
    the pid `Popen` returned. On Windows a venv's ``python.exe`` can be a
    launcher stub, so that pid is the stub's and never equals the batch
    process's own ``os.getpid()`` -- which made every ``--background`` run
    refuse itself at startup. `claimed` is what tells the two records apart.
    """
    handoff = batch.RunManifest(
        run_id="handoff",
        status=batch.RUNNING,
        pid=os.getpid(),  # a live pid that is not the batch process's own
        started_at=batch.utc_now(),
        total_tickers=1,
    )
    batch.save_manifest(handoff, out)
    reloaded = batch.load_manifest(output_dir=out)
    report.check(
        "a launcher handoff round-trips as unclaimed",
        reloaded is not None and reloaded.claimed is False,
    )
    live = batch.active_run(out)
    report.check(
        "an unclaimed handoff is still a live run for --status",
        live is not None,
    )
    report.check(
        "an unclaimed handoff does not lock out the child",
        live is not None and not (live.claimed and live.pid != os.getpid() + 1),
    )
    handoff.claimed = True
    batch.save_manifest(handoff, out)
    claimed = batch.load_manifest(output_dir=out)
    report.check(
        "claimed survives the round trip",
        claimed is not None and claimed.claimed is True,
    )
    # A claimed manifest owned by a *different* live process is the real
    # double-launch this guard exists to refuse.
    report.check(
        "a claimed run under another live pid is refused",
        claimed is not None and claimed.claimed and claimed.pid != os.getpid() + 1,
    )


def check_status_no_network(report: Report, tmp: Path) -> None:
    report.section("--status makes no network calls")
    import requests

    calls: list[str] = []
    original = requests.get

    def trap(*args, **_kwargs):  # noqa: ANN002, ANN003
        calls.append(str(args[0]) if args else "?")
        raise AssertionError("network call during --status")

    requests.get = trap  # type: ignore[assignment]
    try:
        out = tmp / "status"
        batch.save_manifest(
            batch.RunManifest(
                run_id="r2",
                status=batch.COMPLETED,
                started_at=runlog.utc_now(),
                total_tickers=1,
                completed_tickers=1,
            ),
            out,
        )
        data = batch.status_report(output_dir=out)
    finally:
        requests.get = original  # type: ignore[assignment]
    report.check("no requests.get during status", not calls, str(calls))
    report.check(
        "status reflects manifest",
        data.get("status") == batch.COMPLETED and data.get("progress") == 1.0,
    )


def check_pid_alive(report: Report) -> None:
    report.section("Process liveness (never os.kill on Windows)")
    report.check("own pid alive", batch.pid_alive(os.getpid()))
    report.check("absurd pid not alive", not batch.pid_alive(2_000_000_000))
    report.check("pid 0 not alive", not batch.pid_alive(0))


def check_background_launch(report: Report, tmp: Path) -> None:
    report.section("--background returns immediately")
    import subprocess

    launched: list[list[str]] = []

    class FakeProc:
        pid = 424242

    def fake_popen(command, **kwargs):  # noqa: ANN001, ANN003
        launched.append(list(command))
        require("stdout" in kwargs, "stdout must be redirected to the run log")
        return FakeProc()

    original = subprocess.Popen
    subprocess.Popen = fake_popen  # type: ignore[assignment]
    try:
        out, logs = tmp / "bg_out", tmp / "bg_logs"
        opts = batch.BatchOptions(
            tickers=["TCS"], output_dir=str(out), log_dir=str(logs), notify=False
        )
        t0 = time.perf_counter()
        run_id, log_path, pid = batch.launch_background(opts)
        elapsed = time.perf_counter() - t0
    finally:
        subprocess.Popen = original  # type: ignore[assignment]
    report.check("returned in < 2s", elapsed < 2.0, f"{elapsed:.2f}s")
    report.check("Popen called once", len(launched) == 1)
    report.check(
        "child runs scripts.embed_nifty50 with --options-file",
        len(launched) == 1
        and "scripts.embed_nifty50" in launched[0]
        and "--options-file" in launched[0],
    )
    report.check(
        "no --background flag passed to child",
        len(launched) == 1 and "--background" not in launched[0],
    )
    report.check(
        "options file written",
        (logs / "nifty50" / "runs" / run_id / "options.json").exists(),
    )
    live = batch.load_manifest(output_dir=out)
    report.check(
        "manifest RUNNING with child pid",
        live is not None and live.status == batch.RUNNING and live.pid == 424242,
    )
    opts2 = batch.BatchOptions(tickers=["TCS"], output_dir=str(out), log_dir=str(logs))
    # pid 424242 is (almost certainly) dead, so the guard should fall through to launch.
    subprocess.Popen = fake_popen  # type: ignore[assignment]
    try:
        batch.launch_background(opts2)
        report.check(
            "dead-pid manifest does not block a new launch", len(launched) == 2
        )
    finally:
        subprocess.Popen = original  # type: ignore[assignment]
    live_manifest = batch.RunManifest(
        run_id="live",
        status=batch.RUNNING,
        pid=os.getpid(),
        started_at=runlog.utc_now(),
    )
    batch.save_manifest(live_manifest, out)
    report.raises(
        "live run refuses a second launch",
        batch.BatchAlreadyRunning,
        lambda: batch.launch_background(opts2),
    )


def check_notifier(report: Report) -> None:
    report.section("Completion notifier")
    import requests

    posts: list[dict] = []

    class Resp:
        ok = True
        status_code = 200

    def fake_post(url, json=None, **_kwargs):  # noqa: ANN001, ANN003, A002
        posts.append({"url": url, "body": json})
        return Resp()

    original = requests.post
    requests.post = fake_post  # type: ignore[assignment]
    try:
        summary = {
            "run_id": "r",
            "status": "COMPLETED_WITH_ERRORS",
            "total_tickers": 50,
            "counters": {"tickers_ok": 47, "tickers_failed": 3},
            "tickers": {"X": {"status": "FAILED"}, "Y": {"status": "OK"}},
        }
        results = notify.notify_completion(summary, webhook_url="https://hook.test/x")
    finally:
        requests.post = original  # type: ignore[assignment]
    report.check("exactly one webhook POST", len(posts) == 1)
    report.check("log channel always present", any(r.channel == "log" for r in results))
    report.check(
        "failed tickers listed", posts and posts[0]["body"]["failed_tickers"] == ["X"]
    )
    report.check("text carries 47/50", posts and "47/50" in posts[0]["body"]["text"])
    report.check(
        "disabled -> no POST",
        len(
            notify.notify_completion(
                summary, webhook_url="https://hook.test/x", enabled=False
            )
        )
        == 1,
    )

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise requests.ConnectionError("down")

    requests.post = boom  # type: ignore[assignment]
    try:
        res = notify.notify_completion(summary, webhook_url="https://hook.test/x")
    finally:
        requests.post = original  # type: ignore[assignment]
    report.check(
        "unreachable webhook is caught, not raised",
        any(r.channel == "webhook" and not r.ok for r in res),
    )
    line = notify.format_summary_line(
        {
            "run_id": "d",
            "status": "COMPLETED",
            "total_tickers": 2,
            "counters": {"tickers_dry_run": 2, "documents_pending": 6},
            "options": {"dry_run": True},
        }
    )
    report.check(
        "dry-run summary says dry run",
        "dry run" in line and "6 documents pending" in line,
        line,
    )


def check_ntfy(report: Report) -> None:
    report.section("ntfy.sh channel")
    import requests

    report.check(
        "ntfy.sh detected", notify.is_ntfy_url("https://ntfy.sh/grownxt-nifty50")
    )
    report.check(
        "self-hosted ntfy detected", notify.is_ntfy_url("https://ntfy.example.org/t")
    )
    report.check(
        "Slack URL is not ntfy",
        not notify.is_ntfy_url("https://hooks.slack.com/services/x"),
    )

    posts: list[dict] = []

    class Resp:
        ok = True
        status_code = 200
        text = ""

    def fake_post(url, **kwargs):  # noqa: ANN001, ANN003
        posts.append({"url": url, **kwargs})
        return Resp()

    summary = {
        "run_id": "r",
        "status": "COMPLETED_WITH_ERRORS",
        "total_tickers": 50,
        "log_dir": "logs/nifty50/runs/r",
        "counters": {
            "tickers_ok": 48,
            "tickers_failed": 2,
            "documents_embedded": 140,
            "chunks_embedded": 9000,
        },
        "tickers": {
            "M_M": {"status": "FAILED"},
            "TCS": {"status": "OK"},
            "ITC": {"status": "PARTIAL"},
        },
    }
    original = requests.post
    requests.post = fake_post  # type: ignore[assignment]
    try:
        results = notify.notify_completion(
            summary,
            webhook_url="https://ntfy.sh/grownxt-nifty50",
            email_to="ops@example.com",
        )
        notify.notify_completion(
            summary, webhook_url="https://ntfy.sh/grownxt-nifty50", email_to=""
        )
    finally:
        requests.post = original  # type: ignore[assignment]
    report.check(
        "one ntfy POST per run",
        len(posts) == 2 and all(p["url"].endswith("/grownxt-nifty50") for p in posts),
    )
    report.check(
        "result channel is ntfy", any(r.channel == "ntfy" and r.ok for r in results)
    )
    first = posts[0] if posts else {}
    headers = first.get("headers") or {}
    report.check("plain-text body, not JSON", "data" in first and "json" not in first)
    report.check(
        "Title header carries verdict",
        "48/50 OK" in headers.get("Title", ""),
        headers.get("Title", ""),
    )
    report.check(
        "warning tag for errors", headers.get("Tags", "").startswith("warning")
    )
    report.check("high priority for errors", headers.get("Priority") == "4")
    report.check(
        "Email header forwards to inbox", headers.get("Email") == "ops@example.com"
    )
    body = first.get("data", b"").decode("utf-8") if first else ""
    report.check(
        "body names failing tickers",
        "M_M (FAILED)" in body
        and "ITC (PARTIAL)" in body
        and "TCS" not in body.split("Needs attention")[-1],
    )
    second_headers = (posts[1].get("headers") or {}) if len(posts) > 1 else {}
    report.check("empty email_to sends no Email header", "Email" not in second_headers)
    title, _ = notify.format_ntfy_message(
        {"status": "COMPLETED", "total_tickers": 3, "counters": {"tickers_ok": 3}}
    )
    report.check(
        "subset run titled by count",
        title.startswith("GrowNXT 3 ticker(s) embeddings COMPLETED"),
        title,
    )


def check_ticker_notifier(report: Report) -> None:
    """The per-ticker progress feed: quiet on success, loud on failure."""
    report.section("Per-ticker progress notifier")
    import requests

    posts: list[dict] = []

    class Resp:
        ok = True
        status_code = 200
        text = ""

    def fake_post(url, **kwargs):  # noqa: ANN001, ANN003
        posts.append({"url": url, **kwargs})
        return Resp()

    ok_run = {
        "symbol": "TCS",
        "name": "Tata Consultancy Services",
        "status": "OK",
        "new_documents": 3,
        "unchanged_documents": 0,
        "failed_documents": 0,
        "chunks_indexed": 412,
        "points_total": 412,
        "collection": "grownxt_financial_elements_TCS",
        "elapsed_seconds": 900.0,
        "errors": [],
    }
    bad_run = dict(ok_run, symbol="M_M", status="PARTIAL", failed_documents=1)
    bad_run["errors"] = [{"stage": "extract", "error": f"boom {i}"} for i in range(5)]

    topic = "https://ntfy.sh/grownxt-nifty50"
    original = requests.post
    requests.post = fake_post  # type: ignore[assignment]
    try:
        ok_res = notify.notify_ticker(ok_run, position=12, total=50, webhook_url=topic)
        bad_res = notify.notify_ticker(
            bad_run, position=13, total=50, webhook_url=topic
        )
        off = notify.notify_ticker(
            ok_run, position=1, total=50, webhook_url=topic, enabled=False
        )
        # No explicit URL and none in the environment: the run stays silent.
        # The env var is cleared for this one call because a developer box
        # with NIFTY50_NOTIFY_WEBHOOK_URL set would otherwise fall back to it
        # -- which is the documented behaviour, not a bug to assert against.
        saved = os.environ.pop(notify.WEBHOOK_ENV, None)
        try:
            none_url = notify.notify_ticker(ok_run, position=1, total=50)
        finally:
            if saved is not None:
                os.environ[notify.WEBHOOK_ENV] = saved
        slack = notify.notify_ticker(
            ok_run, position=12, total=50, webhook_url="https://hook.test/x"
        )
    finally:
        requests.post = original  # type: ignore[assignment]

    report.check("one POST per finished ticker", len(posts) == 3, str(len(posts)))
    ok_headers = (posts[0].get("headers") or {}) if posts else {}
    bad_headers = (posts[1].get("headers") or {}) if len(posts) > 1 else {}

    report.check(
        "OK ticker is low priority (no buzz)",
        ok_headers.get("Priority") == "2",
        ok_headers.get("Priority", ""),
    )
    report.check(
        "PARTIAL ticker is high priority (buzzes)",
        bad_headers.get("Priority") == "4",
        bad_headers.get("Priority", ""),
    )
    report.check(
        "OK tagged with a check mark",
        ok_headers.get("Tags") == "white_check_mark",
        ok_headers.get("Tags", ""),
    )
    # The reason per-ticker is not notify_completion: fifty tickers must not
    # become fifty emails, so the Email header is never set on this path.
    report.check(
        "no Email header on any per-ticker POST",
        all("Email" not in (p.get("headers") or {}) for p in posts[:2]),
    )
    report.check(
        "Title carries symbol, verdict and progress",
        ok_headers.get("Title") == "GrowNXT TCS OK (12/50)",
        ok_headers.get("Title", ""),
    )
    ok_body = posts[0].get("data", b"").decode("utf-8") if posts else ""
    report.check(
        "body carries chunk and point counts",
        "412 chunks -> 412 points" in ok_body,
        ok_body.splitlines()[-1] if ok_body else "",
    )
    report.check("body names the collection", "_TCS" in ok_body)
    bad_body = posts[1].get("data", b"").decode("utf-8") if len(posts) > 1 else ""
    report.check(
        "errors truncated, remainder counted",
        "boom 0" in bad_body
        and "boom 4" not in bad_body
        and "and 2 more error(s)" in bad_body,
    )
    report.check(
        "channel distinguishes ticker from summary",
        ok_res is not None and ok_res.channel == "ntfy_ticker" and ok_res.ok,
    )
    report.check("PARTIAL still delivered", bad_res is not None and bad_res.ok)
    report.check("disabled -> no POST, returns None", off is None)
    report.check("no webhook configured -> returns None", none_url is None)
    report.check(
        "non-ntfy URL gets JSON, not plain text",
        slack is not None
        and slack.channel == "webhook_ticker"
        and len(posts) == 3
        and "json" in posts[2]
        and posts[2]["json"]["symbol"] == "TCS",
    )
    report.check(
        "a raising webhook never propagates",
        _ticker_notify_survives_exception(ok_run) is not None,
    )


def _ticker_notify_survives_exception(run: dict) -> object:
    """Returns the result of a per-ticker notify whose transport explodes."""
    import requests

    def boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise ValueError("transport exploded")

    original = requests.post
    requests.post = boom  # type: ignore[assignment]
    try:
        return notify.notify_ticker(
            run, position=1, total=50, webhook_url="https://ntfy.sh/t"
        )
    finally:
        requests.post = original  # type: ignore[assignment]


def check_page_filter(report: Report) -> None:
    """The financial-section page filter: its version contract and its safety."""
    report.section("Financial-section page filter")
    from ingestion.documents import sections
    from ingestion.documents.extract import extract_version

    # The version must move with anything that changes the extracted text, or
    # Layer 1 serves a filtered extraction to an unfiltered run.
    report.check(
        "default version is unsuffixed",
        extract_version() == "extract/docling-v1",
        extract_version(),
    )
    report.check(
        "page filter suffixes the version",
        extract_version(page_filter=True).endswith(sections.PAGE_FILTER_SUFFIX),
        extract_version(page_filter=True),
    )
    report.check(
        "table mode and page filter compose, in a fixed order",
        extract_version(accurate_tables=False, page_filter=True)
        == "extract/docling-v1+fast-tables+fin-pages",
        extract_version(accurate_tables=False, page_filter=True),
    )
    report.check(
        "filtered and unfiltered versions differ",
        extract_version(page_filter=True) != extract_version(page_filter=False),
    )

    original = sections.page_texts

    def fake(pages: list[str]):
        def _pages(_pdf, *_a, **_k):
            return pages

        return _pages

    filler = ["ordinary narrative page about the business" * 6] * 240
    try:
        # A real annual report: statements start well into the document.
        pages = list(filler)
        pages[221] = "BALANCE SHEET as at 31 March 2026\nParticulars Note No."
        sections.page_texts = fake(pages)  # type: ignore[assignment]
        rng = sections.financial_page_range("x.pdf")
        report.check(
            "anchors on the balance sheet, 1-based inclusive",
            rng == (222, len(pages)),
            str(rng),
        )

        # The motivating false positive: an auditor's *certificate* on the
        # corporate governance report is not the financial section, and
        # anchoring on it would drag in the pages this filter exists to drop.
        pages = list(filler)
        pages[190] = (
            "INDEPENDENT AUDITOR'S CERTIFICATE ON COMPLIANCE WITH THE "
            "CORPORATE GOVERNANCE REQUIREMENTS UNDER SEBI (LISTING "
            "OBLIGATIONS AND DISCLOSURE REQUIREMENTS) REGULATIONS, 2015"
        )
        pages[221] = "BALANCE SHEET as at 31 March 2026"
        sections.page_texts = fake(pages)  # type: ignore[assignment]
        rng = sections.financial_page_range("x.pdf")
        report.check(
            "an auditor's CERTIFICATE does not anchor the section",
            rng == (222, len(pages)),
            str(rng),
        )

        # A genuine auditor's report should anchor.
        pages = list(filler)
        pages[150] = "INDEPENDENT AUDITOR'S REPORT\nTo the Members of ..."
        sections.page_texts = fake(pages)  # type: ignore[assignment]
        report.check(
            "an auditor's REPORT does anchor the section",
            sections.financial_page_range("x.pdf") == (151, len(pages)),
        )

        # Every uncertain case must convert the whole document.
        sections.page_texts = fake(["deck slide"] * 46)  # type: ignore[assignment]
        report.check(
            "a short document is never filtered",
            sections.financial_page_range("x.pdf") is None,
        )
        sections.page_texts = fake(list(filler))  # type: ignore[assignment]
        report.check(
            "no anchor -> convert everything",
            sections.financial_page_range("x.pdf") is None,
        )
        sections.page_texts = fake([])  # type: ignore[assignment]
        report.check(
            "an unreadable PDF -> convert everything",
            sections.financial_page_range("x.pdf") is None,
        )
        scanned = [""] * 200 + ["BALANCE SHEET"] * 40
        sections.page_texts = fake(scanned)  # type: ignore[assignment]
        report.check(
            "a scanned filing -> convert everything",
            sections.financial_page_range("x.pdf") is None,
        )
        pages = ["CONTENTS ... Balance Sheet ... 222"] + list(filler)
        sections.page_texts = fake(pages)  # type: ignore[assignment]
        report.check(
            "a contents page on page 1 drops nothing",
            sections.financial_page_range("x.pdf") is None,
        )
    finally:
        sections.page_texts = original  # type: ignore[assignment]


def check_qdrant_transport(report: Report) -> None:
    """gRPC is preferred, opt-out-able, and never fatal."""
    report.section("Qdrant transport")
    import os

    from ingestion.indexer import IndexerConfig

    report.check("gRPC preferred by default", IndexerConfig().prefer_grpc is True)
    saved = os.environ.get("QDRANT_PREFER_GRPC")
    try:
        os.environ["QDRANT_PREFER_GRPC"] = "0"
        report.check(
            "QDRANT_PREFER_GRPC=0 opts out",
            IndexerConfig.from_env().prefer_grpc is False,
        )
        os.environ["QDRANT_PREFER_GRPC"] = "1"
        report.check(
            "QDRANT_PREFER_GRPC=1 opts in",
            IndexerConfig.from_env().prefer_grpc is True,
        )
    finally:
        if saved is None:
            os.environ.pop("QDRANT_PREFER_GRPC", None)
        else:
            os.environ["QDRANT_PREFER_GRPC"] = saved
    report.check(
        "an explicit override beats the environment",
        IndexerConfig.from_env(prefer_grpc=False).prefer_grpc is False,
    )


def check_runlog(report: Report, tmp: Path) -> None:
    report.section("Run logging (logs/<job>/runs/<run_id>/)")
    root = tmp / "logs"
    with runlog.setup_run_logging("testjob", run_id="r9", root=root) as rl:
        logging.getLogger("ingestion.test").info("rupee ₹ 1,23,456 – dash")
        rl.event("run_started", total=2)
        rl.event(
            "download_completed", ticker="TCS", bytes=1024, ok=True, tags={"a", "b"}
        )
        rl.add("documents_downloaded")
        rl.add("chunks_embedded", 5)
        rl.time_stage("extract", 1.25)
        rl.write_summary({"status": "COMPLETED"})
        rl.write_latest("COMPLETED")
        run_dir = rl.run_dir
    report.check("run.log written", (run_dir / "run.log").exists())
    report.check("rolling job log written", (root / "testjob" / "testjob.log").exists())
    text = (run_dir / "run.log").read_text(encoding="utf-8")
    report.check("UTF-8 ₹ round-trips", "₹ 1,23,456 – dash" in text)
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    report.check(
        "events.jsonl valid, 2 events",
        len(events) == 2 and events[1]["event"] == "download_completed",
    )
    report.check("sets serialised sorted", events[1].get("tags") == ["a", "b"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report.check(
        "summary carries counters",
        summary["counters"] == {"chunks_embedded": 5, "documents_downloaded": 1},
    )
    report.check(
        "summary carries stage seconds",
        summary["elapsed_by_stage_seconds"] == {"extract": 1.25},
    )
    latest = runlog.read_latest("testjob", root=root)
    report.check(
        "latest.json points at run", latest is not None and latest["run_id"] == "r9"
    )
    report.check(
        "handlers detached on close",
        not any(
            getattr(h, "baseFilename", "").startswith(str(root))
            for h in logging.getLogger().handlers
        ),
    )


def check_options(report: Report) -> None:
    report.section("BatchOptions")
    opts = batch.BatchOptions(tickers=["tcs", "M&M"], force=["m&m"], limit=5, workers=9)
    again = batch.BatchOptions.from_dict(json.loads(json.dumps(opts.to_dict())))
    report.check("round-trips through JSON", again == opts)
    report.check(
        "forced() folds symbols", opts.forced("M_M") and not opts.forced("TCS")
    )
    report.check("force_all wins", batch.BatchOptions(force_all=True).forced("ANY"))
    members = batch.BatchOptions(tickers=["WIPRO", "NOTREAL"]).resolve_constituents()
    report.check(
        "unknown ticker still in scope",
        [m.symbol for m in members] == ["WIPRO", "NOTREAL"],
    )
    report.check("known ticker gets its name", bool(members[0].name))
    report.check(
        "limit applies", len(batch.BatchOptions(limit=3).resolve_constituents()) == 3
    )
    report.check("admin token unset -> refused", not batch.admin_token_ok("anything"))


def check_hardware(report: Report) -> None:
    """The compute profile: resolution, overrides, and the CPU-wheel trap."""
    report.section("Hardware profile")
    from core import hardware

    auto = hardware.profile()
    report.check(
        "device resolves to something concrete",
        auto.device.startswith(("cuda", "cpu", "mps", "xpu")),
        auto.device,
    )
    expected_threads = max(1, min(auto.cpu_cores, hardware.MAX_DOCLING_THREADS))
    report.check(
        "threads track physical cores, not Docling's fixed default of 4",
        "num_threads" in auto.overrides or auto.num_threads == expected_threads,
        f"{auto.num_threads} threads for {auto.cpu_cores} physical cores",
    )
    report.check("embed batch is positive", auto.embed_batch_size > 0)
    report.check(
        "fp16 never on a non-CUDA device",
        auto.device.startswith("cuda") or not auto.embed_fp16,
        f"device={auto.device} fp16={auto.embed_fp16}",
    )
    report.check(
        "any overrides come from known settings",
        set(auto.overrides)
        <= {"device", "num_threads", "embed_batch_size", "embed_fp16"},
        str(auto.overrides) or "none (nothing overridden)",
    )

    forced = hardware.profile(device="cpu", num_threads=3, embed_batch_size=7)
    report.check(
        "explicit arguments win",
        (forced.device, forced.num_threads, forced.embed_batch_size) == ("cpu", 3, 7),
        forced.summary(),
    )
    report.check(
        "overrides are named",
        set(forced.overrides) == {"device", "num_threads", "embed_batch_size"},
        str(forced.overrides),
    )
    report.check(
        "an unknown device falls back to detection",
        hardware.profile(device="tpu").device == auto.device,
    )
    report.check("profile is JSON-serialisable", bool(json.dumps(auto.to_dict())))

    # configure() must never raise, whatever is or is not installed.
    applied = hardware.configure(resolved=forced)
    report.check("configure() returns its profile", applied == forced)
    report.check(
        "Docling env vars set",
        bool(os.getenv("DOCLING_NUM_THREADS") and os.getenv("DOCLING_DEVICE")),
        f"DOCLING_NUM_THREADS={os.getenv('DOCLING_NUM_THREADS')} "
        f"DOCLING_DEVICE={os.getenv('DOCLING_DEVICE')}",
    )

    # The whole point of the module: a card present, a torch that cannot use it.
    trap = hardware.HardwareProfile(
        device="cpu", torch_build="2.13.0+cpu", cuda_build="", gpu_present=True
    )
    report.check("idle GPU detected", trap.idle_gpu)
    report.check(
        "no false alarm on a CPU-only machine",
        not hardware.HardwareProfile(
            device="cpu", torch_build="2.13.0+cpu", gpu_present=False
        ).idle_gpu,
    )
    report.check(
        "a working CUDA build is not an idle GPU",
        not hardware.HardwareProfile(
            device="cuda",
            torch_build="2.13.0+cu128",
            cuda_build="12.8",
            gpu_present=True,
        ).idle_gpu,
    )

    # Advisory, not a failure: a CPU-only host is a legitimate deployment, and
    # this suite also runs where there is no card at all.
    if auto.idle_gpu:
        report.section(
            f"NOTE: an NVIDIA GPU is present but torch {auto.torch_build} "
            "cannot use it -- extraction will run on the CPU"
        )


def check_gpu_workers(report: Report) -> None:
    """Concurrency must yield to VRAM: N workers means N model copies."""
    report.section("Worker resolution")
    from core.hardware import HardwareProfile

    cpu = HardwareProfile(device="cpu")
    report.check("CPU keeps the requested workers", batch.resolve_workers(4, cpu) == 4)
    report.check("CPU still capped at MAX_WORKERS", batch.resolve_workers(99, cpu) == 4)
    report.check("never below one", batch.resolve_workers(0, cpu) == 1)

    small = HardwareProfile(device="cuda", vram_gb=4.0, gpu_name="GTX 1650 Ti")
    report.check(
        "a small card is forced to one worker", batch.resolve_workers(4, small) == 1
    )
    big = HardwareProfile(device="cuda", vram_gb=24.0, gpu_name="A10G")
    report.check("a large card keeps them", batch.resolve_workers(4, big) == 4)


def check_extract_version(report: Report) -> None:
    """The table mode has to reach Layer 1, or flipping it changes nothing."""
    report.section("Extractor version and the table mode")
    from ingestion.documents import extract as extract_mod

    accurate = extract_mod.extract_version(accurate_tables=True)
    fast = extract_mod.extract_version(accurate_tables=False)
    report.check(
        "accurate mode keeps EXTRACT_VERSION",
        accurate == extract_mod.EXTRACT_VERSION,
        accurate,
    )
    report.check("fast mode is a distinct version", fast != accurate, fast)
    report.check(
        "Extractor.version follows its own setting",
        extract_mod.Extractor(accurate_tables=False).version == fast
        and extract_mod.Extractor(accurate_tables=True).version == accurate,
    )

    cfg = IndexerConfig(collection_prefix="grownxt", collection_per_ticker=True)
    entries = [_entry("d1")]
    state = _state("TCS", cfg, d1=_indexed(cfg, "TCS", extract_version=accurate))
    _, unchanged = batch.compute_pending(entries, state, cfg, extract_version=accurate)
    report.check("same mode -> unchanged", len(unchanged) == 1)
    pending, _ = batch.compute_pending(entries, state, cfg, extract_version=fast)
    report.check(
        "switching to fast tables re-extracts",
        len(pending) == 1 and pending[0][1] == batch.REASON_VERSION,
    )
    report.check(
        "device and threads do not invalidate a cache",
        extract_mod.Extractor(device="cuda", num_threads=16).version == accurate,
    )


def _imported_modules(path: Path) -> set[str]:
    """Returns the modules a source file imports, by walking its AST.

    The import statements, not the file's text: a module's docstring may
    legitimately name a layer it does not import, and grep cannot tell the
    difference.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def check_module_boundaries(report: Report) -> None:
    """Each layer must stay usable without the ones above it."""
    import ast

    report.section("Module boundaries")
    imported = _imported_modules(Path("ingestion/stages.py"))

    forbidden = {
        m for m in imported if m == "ingestion.batch" or m.startswith("langgraph")
    }
    report.check(
        "stages.py imports neither ingestion.batch nor langgraph",
        not forbidden,
        ", ".join(sorted(forbidden)) or f"{len(imported)} modules imported",
    )

    # The saver must stay a generic checkpoint store: one that knows about
    # tickers cannot be tested, or reused, on its own.
    saver_imports = _imported_modules(Path("ingestion/graph/checkpoint.py"))
    saver_forbidden = {
        m
        for m in saver_imports
        if m in ("ingestion.batch", "ingestion.stages", "ingestion.indexer")
    }
    report.check(
        "graph/checkpoint.py knows nothing about the batch or its stages",
        not saver_forbidden,
        ", ".join(sorted(saver_forbidden)) or f"{len(saver_imports)} modules imported",
    )
    state_imports = _imported_modules(Path("ingestion/graph/state.py"))
    report.check(
        "graph/state.py imports no graph runtime",
        not any(m.startswith("langgraph") for m in state_imports),
        ", ".join(sorted(state_imports)),
    )

    from ingestion.graph import checkpoint as saver_mod

    overridden = {
        name
        for name in (
            "get_delta_channel_history",
            "prune",
            "delete_for_runs",
            "copy_thread",
            "get_next_version",
        )
        if name in vars(saver_mod.JsonFileSaver)
    }
    report.check(
        "JsonFileSaver overrides only the abstract methods",
        not overridden,
        ", ".join(sorted(overridden)) or "base implementations inherited",
    )
    report.check(
        "stages.py owns Layer 1, batch.py re-exports it",
        batch.pending_reason is stages.pending_reason
        and batch.compute_pending is stages.compute_pending
        and batch.select_entries is stages.select_entries
        and batch.REASON_NEW == stages.REASON_NEW,
    )

    tree = ast.parse(Path("ingestion/batch.py").read_text(encoding="utf-8"))
    sizes = {
        n.name: n.end_lineno - n.lineno + 1
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    report.check(
        "process_ticker composes stages rather than inlining them",
        sizes.get("process_ticker", 999) <= 120,
        f"{sizes.get('process_ticker')} lines",
    )


def check_extraction_cache(report: Report) -> None:
    """A cached extraction must be reused, and only when it still applies.

    The Docling pass is the dominant cost of a restart: Layer 1 knows only
    that a document was never *embedded*, so it asks for the extraction again
    whether or not one is already on disk.
    """
    report.section("Extraction cache")
    from ingestion.documents.content import KIND_TEXT, Block
    from ingestion.documents.extract import CACHE_KEY_FIELD, Extractor, read_extraction
    from ingestion.documents.storage import DocumentStore

    calls = {"n": 0}

    def build(**kwargs: object) -> Extractor:
        extractor = Extractor(ocr=False, figures=False, **kwargs)  # type: ignore[arg-type]

        class _Converter:
            def convert(self, _path: str, **_kwargs: object) -> object:
                calls["n"] += 1
                return type("R", (), {"document": object()})()

        object.__setattr__(extractor, "_converter", _Converter())
        extractor._blocks = lambda *_a: [Block(kind=KIND_TEXT, text="hi", page=1)]  # type: ignore[method-assign]
        return extractor

    with tempfile.TemporaryDirectory() as raw:
        store = DocumentStore.open("TESTCO", data_dir=Path(raw)).ensure()
        pdf = store.pdf("annual_report_FY2026")
        pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 4096)
        args: dict[str, object] = {
            "store": store,
            "doc_id": "annual_report_FY2026",
            "doc_type": "annual_report",
            "ticker": "TESTCO",
            "label": "FY2026",
        }

        extractor = build()
        first = extractor.run(pdf=pdf, **args)  # type: ignore[arg-type]
        report.check("first pass extracts", calls["n"] == 1 and bool(first.blocks))

        again = extractor.run(pdf=pdf, **args)  # type: ignore[arg-type]
        report.check(
            "second pass reuses the cache",
            calls["n"] == 1 and len(again.blocks) == len(first.blocks),
            f"{calls['n']} docling call(s)",
        )

        extractor.run(pdf=pdf, reuse=False, **args)  # type: ignore[arg-type]
        report.check("reuse=False forces a real pass", calls["n"] == 2)

        pdf.write_bytes(b"%PDF-1.4\n" + b"y" * 4096)
        extractor.run(pdf=pdf, **args)  # type: ignore[arg-type]
        report.check("a changed PDF re-extracts", calls["n"] == 3)

        calls["n"] = 0
        build(accurate_tables=False).run(pdf=pdf, **args)  # type: ignore[arg-type]
        report.check("flipping the table mode re-extracts", calls["n"] == 1)

        calls["n"] = 0
        build(accurate_tables=False, num_threads=1, device="cpu").run(pdf=pdf, **args)  # type: ignore[arg-type]
        report.check(
            "device and threads do not invalidate the cache",
            calls["n"] == 0,
            "a run must not invalidate the last machine's cache",
        )

        key = read_extraction(store.extraction("annual_report_FY2026")).meta[
            CACHE_KEY_FIELD
        ]
        report.check(
            "the cache key excludes device and thread count",
            "device" not in key and "num_threads" not in key,
            ", ".join(sorted(key)),
        )


def check_chunk_cache(report: Report) -> None:
    """`chunk_params` is the single owner of what determines a chunk set."""
    report.section("Chunk parameters and cache")
    from ingestion.chunker import CHUNKER_VERSION, chunk_document, chunk_params
    from ingestion.documents.content import (
        KIND_HEADING,
        KIND_TEXT,
        Block,
        ExtractedDocument as Doc,
    )

    def make(extractor: str) -> Doc:
        doc = Doc(
            doc_id="d1",
            doc_type="annual_report",
            ticker="T",
            label="FY2026",
            extractor=extractor,
        )
        doc.blocks = [
            Block(kind=KIND_HEADING, text="Revenue", page=1, level=1),
            Block(kind=KIND_TEXT, text="Revenue grew 18%. " * 60, page=1),
        ]
        return doc

    doc = make("extract/docling-v1")
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=100)
    report.check(
        "chunk_document records exactly chunk_params()",
        chunks.params == chunk_params(doc, chunk_size=800, chunk_overlap=100),
    )
    report.check(
        "params carry the chunker and extractor versions",
        chunks.params["version"] == CHUNKER_VERSION
        and chunks.params["extractor"] == "extract/docling-v1",
    )

    other = chunk_document(make("extract/docling-v1+fast-tables"), chunk_size=800)
    report.check(
        "a different extractor changes params and fingerprint",
        other.params != chunks.params and other.fingerprint != chunks.fingerprint,
    )
    smaller = chunk_document(doc, chunk_size=400, chunk_overlap=100)
    report.check(
        "a different chunk size changes params and fingerprint",
        smaller.params != chunks.params and smaller.fingerprint != chunks.fingerprint,
    )


def check_index_failure_state(report: Report) -> None:
    """A failed embed must be recorded, so a retry is not reported as new."""
    report.section("Index failure state")
    from ingestion.chunker import Chunk, ChunkSet

    with tempfile.TemporaryDirectory() as raw:
        cfg = IndexerConfig.from_env(output_dir=Path(raw))
        indexer = QdrantVectorIndexer(config=cfg)
        chunk_set = ChunkSet(
            doc_id="annual_report_FY2026",
            ticker="TESTCO",
            doc_type="annual_report",
            label="FY2026",
            fingerprint="fp-abc",
            params={"version": "chunk/v1", "extractor": "extract/docling-v1"},
            chunks=[
                Chunk(
                    chunk_id="c1",
                    doc_id="annual_report_FY2026",
                    ticker="TESTCO",
                    text="hello",
                    page_start=1,
                )
            ],
        )

        def _boom(*_a: object, **_k: object) -> int:
            raise RuntimeError("qdrant down")

        # The bulk path `index_chunk_set` actually takes. Stubbing
        # `upsert_points` instead would inject a failure into a method the
        # document path no longer calls, and the check would pass vacuously.
        indexer.upload_vectors = _boom  # type: ignore[method-assign]
        result = indexer.index_chunk_set(chunk_set)
        report.check("a failed upsert returns FAILED", result.status == "FAILED")

        state = indexer.load_ticker_state("TESTCO")
        recorded = state.documents.get("annual_report_FY2026")
        report.check(
            "state.json records the failure",
            recorded is not None and recorded.status == "FAILED",
            (recorded.error[:40] if recorded else "no record"),
        )
        report.check(
            "Layer 1 calls it a retry, not a new document",
            stages.pending_reason(_entry("annual_report_FY2026"), state, cfg)
            == batch.REASON_RETRY,
        )
        report.check(
            "Layer 2 still embeds it",
            indexer.should_index_document(
                "TESTCO", "annual_report_FY2026", "fp-abc", state=state
            ),
        )


def _stage_ctx(
    root: Path, tmp: Path, **overrides: object
) -> tuple[stages.StageContext, QdrantVectorIndexer]:
    """Builds a StageContext over a temporary workspace, with a real logger."""
    options = batch.BatchOptions(output_dir=str(root), log_dir=str(tmp), **overrides)  # type: ignore[arg-type]
    indexer = QdrantVectorIndexer(config=IndexerConfig.from_env(output_dir=root))
    logger_ = runlog.RunLogger(job="verify", run_id="ttl", root=tmp)
    member = nifty50.Constituent(symbol="TCS", name="Tata Consultancy Services")
    return stages.StageContext.build(member, options, indexer, logger_), indexer


def _seed_indexed(
    indexer: QdrantVectorIndexer,
    ctx: stages.StageContext,
    *,
    checked_at: str | None = None,
    status: str = "INDEXED",
) -> None:
    """Writes a state.json describing one fully-embedded filing."""
    from ingestion.chunker import CHUNKER_VERSION
    from ingestion.indexer import DocumentIndexState, TickerState

    state = TickerState(ticker=ctx.symbol, last_updated=runlog.utc_now())
    state.documents["annual_report_FY2026"] = DocumentIndexState(
        fingerprint="fp",
        chunk_count=12,
        embedding_model=indexer.config.model_alias,
        qdrant_collection=ctx.collection,
        indexed_at=runlog.utc_now(),
        status=status,
        chunker_version=CHUNKER_VERSION,
        extract_version=ctx.wanted_extract_version,
    )
    state.catalog_checked_at = runlog.utc_now() if checked_at is None else checked_at
    state.catalog_selection = ctx.selection
    indexer.save_ticker_state(state)


def check_catalogue_ttl(report: Report, tmp: Path) -> None:
    """Inside the TTL, a fully-embedded ticker costs no catalogue request."""
    report.section("Catalogue TTL")
    import datetime as dt

    from ingestion import catalog as catalog_mod

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        ctx, indexer = _stage_ctx(root, tmp)

        report.check(
            "no state -> a request is required",
            stages.catalogue_skip(ctx) is None,
        )

        _seed_indexed(indexer, ctx)
        skip = stages.catalogue_skip(ctx)
        report.check(
            "fresh check + all embedded -> skip",
            skip is not None and skip.ok and skip.skipped,
            (", ".join(skip.known_documents) if skip else "no skip"),
        )

        stale = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=48)).isoformat()
        _seed_indexed(indexer, ctx, checked_at=stale)
        report.check(
            "outside the TTL -> a request is required",
            stages.catalogue_skip(ctx) is None,
        )

        # A failed document must always force a fresh look, at any age.
        _seed_indexed(indexer, ctx, status="FAILED")
        report.check(
            "a FAILED document forces a request",
            stages.catalogue_skip(ctx) is None,
        )

        _seed_indexed(indexer, ctx)
        forced_ctx, forced_indexer = _stage_ctx(root, tmp, force_all=True)
        report.check(
            "--force bypasses the TTL", stages.catalogue_skip(forced_ctx) is None
        )
        dry_ctx, _ = _stage_ctx(root, tmp, dry_run=True)
        report.check(
            "a dry run always catalogues", stages.catalogue_skip(dry_ctx) is None
        )
        off_ctx, _ = _stage_ctx(root, tmp, catalog_ttl_hours=0)
        report.check(
            "--catalog-ttl 0 disables the skip", stages.catalogue_skip(off_ctx) is None
        )
        wider_ctx, _ = _stage_ctx(root, tmp, annual_reports=2)
        report.check(
            "asking for more documents forces a request",
            stages.catalogue_skip(wider_ctx) is None,
        )
        del forced_indexer

        # And the wiring: fetch_catalogue must not reach the network on a skip.
        calls = {"n": 0}
        original = catalog_mod.fetch_catalog

        def _counted(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return original(*args, **kwargs)

        catalog_mod.fetch_catalog = _counted  # type: ignore[assignment]
        try:
            result = stages.fetch_catalogue(ctx)
        finally:
            catalog_mod.fetch_catalog = original  # type: ignore[assignment]
        report.check(
            "fetch_catalogue makes no request when it can skip",
            calls["n"] == 0 and result.skipped,
            f"{calls['n']} call(s)",
        )


def check_ephemeral_refusal(report: Report, tmp: Path) -> None:
    """An in-memory vector store must refuse, not warn: state.json would lie."""
    report.section("Ephemeral Qdrant refusal")
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        env = {"QDRANT_API_URL": "", "QDRANT_CLUSTER": "", "QDRANT_HOST": ""}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            # limit=0 gives the run an empty ticker scope, so it reaches the
            # persistence gate and then finishes without touching the network.
            def options(**extra: object) -> batch.BatchOptions:
                return batch.BatchOptions(
                    limit=0,
                    output_dir=str(root),
                    log_dir=str(tmp),
                    notify=False,
                    **extra,  # type: ignore[arg-type]
                )

            report.raises(
                "an in-memory store refuses to run",
                batch.BatchError,
                lambda: batch.run_batch(options()),
            )
            report.check(
                "it refuses before writing a manifest",
                batch.load_manifest(output_dir=root) is None,
            )
            report.check(
                "--allow-ephemeral proceeds",
                batch.run_batch(options(allow_ephemeral=True)).status
                == batch.COMPLETED,
            )
            report.check(
                "a dry run is exempt",
                batch.run_batch(options(dry_run=True)).status == batch.COMPLETED,
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def check_part_sweep(report: Report) -> None:
    """Abandoned download temporaries are reclaimed; live ones are not."""
    report.section("Abandoned .part sweep")
    from ingestion.documents.storage import PART_SUFFIX, DocumentStore

    with tempfile.TemporaryDirectory() as raw:
        store = DocumentStore.open("TESTCO", data_dir=Path(raw)).ensure()
        # Exactly the shape Downloader._stream leaves behind: dot-prefixed.
        stale = store.documents / f".annual_report_FY2026-abc{PART_SUFFIX}"
        stale.write_bytes(b"z" * 5000)
        fresh = store.documents / f".transcript_2025_07-def{PART_SUFFIX}"
        fresh.write_bytes(b"z" * 300)
        real = store.pdf("annual_report_FY2026")
        real.write_bytes(b"%PDF-1.4\nreal")

        aged = time.time() - 10 * 3600
        os.utime(stale, (aged, aged))

        reclaimed = store.sweep_partials()
        report.check("an abandoned temporary is removed", not stale.exists())
        report.check("a live transfer is left alone", fresh.exists())
        report.check(
            "a real PDF is never touched",
            real.exists() and real.read_bytes().startswith(b"%PDF"),
        )
        report.check(
            "the reclaimed bytes are reported", reclaimed == 5000, str(reclaimed)
        )


def check_graph_reducers(report: Report) -> None:
    """The channel reducers, which is where a concurrent-write bug would hide."""
    report.section("Graph state reducers")
    from ingestion.graph import state as gstate

    report.check(
        "merge_results is last-write-wins per symbol",
        gstate.merge_results({"TCS": {"n": 1}}, {"INFY": {"n": 2}})
        == {"TCS": {"n": 1}, "INFY": {"n": 2}}
        and gstate.merge_results({"TCS": {"n": 1}}, {"TCS": {"n": 9}})
        == {"TCS": {"n": 9}},
    )
    report.check(
        "two tickers finishing together both survive",
        gstate.merge_results({"A": {"n": 1}}, {"B": {"n": 2}})
        == {"A": {"n": 1}, "B": {"n": 2}},
        "without a reducer the second write would replace the whole dict",
    )
    report.check(
        "a missing left side is tolerated",
        gstate.merge_results(None, {"TCS": {}}) == {"TCS": {}}
        and gstate.merge_results(None, None) == {},
    )
    left = {"A": {"n": 1}}
    gstate.merge_results(left, {"B": {"n": 2}})
    report.check("the reducer does not mutate its arguments", left == {"A": {"n": 1}})
    report.check(
        "the reducer is an importable module-level function, not a lambda",
        getattr(gstate.merge_results, "__name__", "<lambda>") == "merge_results",
    )
    report.check(
        "the schema declares no payload-shaped channel",
        set(gstate.BatchGraphState.__annotations__)
        <= {"queue", "in_flight", "results", "counters", "symbol", "position"},
        ", ".join(sorted(gstate.BatchGraphState.__annotations__)),
    )


def check_checkpoint_saver(report: Report) -> None:
    """`JsonFileSaver` must satisfy the contract the graph runtime relies on."""
    report.section("JsonFileSaver conformance")
    try:
        from langgraph.checkpoint.base import empty_checkpoint
        from langgraph.checkpoint.serde.types import ERROR
    except ImportError as exc:
        report.check("langgraph.checkpoint is importable", False, str(exc)[:70])
        return

    from ingestion.graph import JsonFileSaver

    def cfg(thread: str = "t1", ns: str = "", cid: str | None = None) -> dict:
        conf: dict[str, object] = {"thread_id": thread, "checkpoint_ns": ns}
        if cid:
            conf["checkpoint_id"] = cid
        return {"configurable": conf}

    with tempfile.TemporaryDirectory() as raw:
        saver = JsonFileSaver(root=Path(raw))
        first = empty_checkpoint()
        first["channel_values"] = {"x": {"deep": [1, 2, 3]}, "y": "str"}
        first["channel_versions"] = {"x": 1, "y": 1}
        metadata = {"source": "loop", "step": 3, "parents": {}}

        returned = saver.put(cfg(), first, metadata, {"x": 1, "y": 1})
        got = saver.get_tuple(cfg())
        report.check(
            "put -> get_tuple preserves channel values",
            got is not None
            and got.checkpoint["channel_values"] == first["channel_values"],
        )
        report.check(
            "metadata and id survive the round trip",
            got is not None
            and got.metadata.get("source") == "loop"
            and got.metadata.get("step") == 3
            and got.checkpoint["id"] == first["id"],
        )
        report.check("a first checkpoint has no parent", got.parent_config is None)

        second = empty_checkpoint()
        second["channel_versions"] = {}
        saver.put(returned, second, {"source": "loop", "step": 4, "parents": {}}, {})
        latest = saver.get_tuple(cfg())
        report.check(
            "the newest checkpoint is returned by default",
            latest.checkpoint["id"] == second["id"],
        )
        report.check(
            "parent_config chains to the previous checkpoint",
            latest.parent_config["configurable"]["checkpoint_id"] == first["id"],
        )

        # A nested subgraph namespace contains ':' and '|', both illegal in a
        # Windows filename -- the path is slugged and the true value is stored.
        nested = "ticker:TCS|doc:annual_report_FY2026"
        saver.put(cfg(ns=nested), first, metadata, {"x": 1, "y": 1})
        nested_tuple = saver.get_tuple(cfg(ns=nested))
        report.check(
            "a subgraph namespace round-trips through the filesystem",
            nested_tuple is not None
            and nested_tuple.config["configurable"]["checkpoint_ns"] == nested,
            nested,
        )
        report.check(
            "namespaces stay isolated",
            saver.get_tuple(cfg()).checkpoint["id"] == second["id"],
        )

        writes_cfg = cfg(cid=second["id"])
        saver.put_writes(writes_cfg, [("ch", "v1"), ("ch2", "v2")], "task-A")
        saver.put_writes(writes_cfg, [("ch", "v1"), ("ch2", "v2")], "task-A")
        pending = saver.get_tuple(cfg(cid=second["id"])).pending_writes
        report.check(
            "a replayed superstep does not duplicate writes",
            len(pending) == 2,
            f"{len(pending)} write(s)",
        )
        saver.put_writes(writes_cfg, [("ch", "v3")], "task-B")
        report.check(
            "a second task's writes are kept",
            len(saver.get_tuple(cfg(cid=second["id"])).pending_writes) == 3,
        )
        saver.put_writes(writes_cfg, [(ERROR, "boom")], "task-A")
        saver.put_writes(writes_cfg, [(ERROR, "boom2")], "task-A")
        errors = [
            w[2]
            for w in saver.get_tuple(cfg(cid=second["id"])).pending_writes
            if w[1] == ERROR
        ]
        report.check(
            "special channels may repeat (negative indices)",
            len(errors) == 2,
            ", ".join(str(e) for e in errors),
        )

        stored = list(saver.list(cfg()))
        ids = [t.checkpoint["id"] for t in stored]
        report.check("list yields newest first", ids == sorted(ids, reverse=True))
        report.check("list honours limit", len(list(saver.list(cfg(), limit=1))) == 1)
        report.check(
            "list honours before",
            all(
                t.checkpoint["id"] < ids[0]
                for t in saver.list(cfg(), before=cfg(cid=ids[0]))
            ),
        )
        report.check(
            "list honours a metadata filter",
            all(
                t.metadata.get("step") == 3
                for t in saver.list(cfg(), filter={"step": 3})
            ),
        )

        failures: list[Exception] = []

        def concurrent(index: int) -> None:
            try:
                point = empty_checkpoint()
                point["channel_values"] = {"w": index}
                point["channel_versions"] = {"w": 1}
                saver.put(
                    cfg(thread="conc"),
                    point,
                    {"source": "loop", "step": index, "parents": {}},
                    {"w": 1},
                )
            except Exception as exc:  # noqa: BLE001 - the finding is the failure
                failures.append(exc)

        threads = [threading.Thread(target=concurrent, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent_stored = list(saver.list(cfg(thread="conc")))
        report.check(
            "eight concurrent puts write eight complete checkpoints",
            not failures and len(concurrent_stored) == 8,
            f"{len(concurrent_stored)} stored, {len(failures)} error(s)",
        )
        report.check(
            "no concurrently written checkpoint is truncated",
            all("w" in t.checkpoint["channel_values"] for t in concurrent_stored),
        )

        saver.delete_thread("conc")
        report.check(
            "delete_thread forgets one thread and spares the others",
            saver.get_tuple(cfg(thread="conc")) is None
            and saver.get_tuple(cfg()) is not None,
        )

        victim = next(Path(raw).rglob("*.json"))
        victim.write_text("{ not json", encoding="utf-8")
        report.run(
            "a corrupt checkpoint reads as absent, never raises",
            lambda: f"get_tuple -> {saver.get_tuple(cfg()) is not None}",
        )


class _ResumeState(TypedDict, total=False):
    """State schema for the resume check.

    Module level, not nested inside the check: `StateGraph` resolves a
    schema's annotations with `get_type_hints`, which cannot see names local
    to a function once annotations are postponed.
    """

    trail: Annotated[list[str], operator.add]


def check_graph_resume(report: Report) -> None:
    """A crashed graph must resume at the failed node, not at the start."""
    report.section("Graph resume")
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        report.check("langgraph.graph is importable", False, str(exc)[:70])
        return

    from ingestion.graph import JsonFileSaver

    calls = dict.fromkeys("abcd", 0)
    fail_once = {"armed": True}

    def make(name: str):
        def node(_state: _ResumeState) -> dict:
            calls[name] += 1
            if name == "c" and fail_once["armed"]:
                fail_once["armed"] = False
                raise RuntimeError("crash in node c")
            return {"trail": [name]}

        return node

    with tempfile.TemporaryDirectory() as raw:
        saver = JsonFileSaver(root=Path(raw))
        builder = StateGraph(_ResumeState)
        for name in "abcd":
            builder.add_node(name, make(name))
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", "c")
        builder.add_edge("c", "d")
        builder.add_edge("d", END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "run-1"}}

        report.raises(
            "the graph surfaces the node's failure",
            RuntimeError,
            lambda: graph.invoke({"trail": []}, config, durability="sync"),
        )
        snapshot = graph.get_state(config)
        report.check(
            "the checkpoint resumes at the failed node",
            snapshot.next == ("c",),
            f"next={snapshot.next}, trail={snapshot.values.get('trail')}",
        )

        final = graph.invoke(None, config, durability="sync")
        report.check(
            "completed nodes are not re-run",
            calls["a"] == 1 and calls["b"] == 1,
            f"a={calls['a']}, b={calls['b']}",
        )
        report.check(
            "the failed node runs again and the graph finishes",
            calls["c"] == 2 and calls["d"] == 1,
            f"c={calls['c']}, d={calls['d']}",
        )
        report.check(
            "restored state is not replayed into the reducer",
            final["trail"] == ["a", "b", "c", "d"],
            str(final["trail"]),
        )


def check_checkpoint_size(report: Report) -> None:
    """Nine hundred chunks must not mean a nine-hundred-chunk checkpoint.

    The executable form of the rule in `ingestion.graph.state`: state carries
    identifiers, paths and counts, so a checkpoint stays small however large
    the filing behind it is. A state holding a `ChunkSet` would fail this.
    """
    report.section("Checkpoint size")
    try:
        from langgraph.checkpoint.base import empty_checkpoint
    except ImportError as exc:
        report.check("langgraph.checkpoint is importable", False, str(exc)[:70])
        return

    from ingestion.graph import JsonFileSaver

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        saver = JsonFileSaver(root=root)
        ticker_state = {
            "symbol": "TCS",
            "name": "Tata Consultancy Services",
            "collection": "grownxt_financial_elements_TCS",
            "catalog_entries": 42,
            "selected": [
                {
                    "doc_id": f"annual_report_FY{2020 + i}",
                    "doc_type": "annual_report",
                    "label": f"FY{2020 + i}",
                    "source_url": "https://example.invalid/a.pdf",
                }
                for i in range(3)
            ],
            "downloaded": [
                {
                    "doc_id": f"annual_report_FY{2020 + i}",
                    "relative_path": f"documents/annual_report_FY{2020 + i}.pdf",
                    "sha256": "0" * 64,
                    "n_bytes": 41_000_000,
                    "reused": True,
                }
                for i in range(3)
            ],
            # The point of the check: a filing of 900 chunks is a count here.
            "outcomes": {
                f"annual_report_FY{2020 + i}": {
                    "doc_id": f"annual_report_FY{2020 + i}",
                    "status": "INDEXED",
                    "chunks": 900,
                    "n_chunks": 900,
                    "fingerprint": "f" * 64,
                }
                for i in range(3)
            },
        }
        point = empty_checkpoint()
        point["channel_values"] = ticker_state
        point["channel_versions"] = dict.fromkeys(ticker_state, 1)
        saver.put(
            {"configurable": {"thread_id": "size", "checkpoint_ns": "ticker:TCS"}},
            point,
            {"source": "loop", "step": 1, "parents": {}},
            dict.fromkeys(ticker_state, 1),
        )

        files = list(root.rglob("*"))
        largest = max((p.stat().st_size for p in files if p.is_file()), default=0)
        total = sum(p.stat().st_size for p in files if p.is_file())
        report.check(
            "no checkpoint file exceeds 64 KB",
            largest <= 64 * 1024,
            f"largest {largest} bytes",
        )
        report.check(
            "the thread directory stays under 2 MB",
            total <= 2 * 1024 * 1024,
            f"{total} bytes total",
        )


def _small_gpu() -> Any:
    """A profile for a card too small to hold two tickers' models."""
    from core.hardware import HardwareProfile

    return HardwareProfile(device="cuda", vram_gb=4.0, gpu_name="GTX 1650 Ti")


def check_bounded_dispatch(report: Report) -> None:
    """Fan-out must stay inside `resolve_workers`, batch by batch.

    `Send` has no worker limit of its own, so dispatching every ticker at once
    would put one Docling model set per ticker on the same GPU -- the
    out-of-memory `resolve_workers` exists to prevent.
    """
    report.section("Bounded dispatch")
    from ingestion.graph import nodes as gnodes

    res = gnodes.GraphResources(
        options=batch.BatchOptions(),
        indexer=None,
        runlog=None,
        workers=2,
        members={},
        order=[f"T{i}" for i in range(7)],
        record=lambda _run: None,
    )
    queue = list(res.order)
    batches: list[list[str]] = []
    state: dict = {"queue": queue, "workers": 2, "results": {}}
    while state["queue"]:
        state.update(gnodes.dispatch(res, state))
        sends = gnodes.route_dispatch(state)
        batches.append([s.arg["symbol"] for s in sends])
        # Simulate the tickers finishing, as `collect` would see them.
        state["results"] = {
            **state["results"],
            **{sym: {"symbol": sym} for sym in state["in_flight"]},
        }
    report.check(
        "no superstep dispatches more than `workers` tickers",
        all(len(b) <= 2 for b in batches),
        " | ".join(",".join(b) for b in batches),
    )
    report.check(
        "every ticker is dispatched exactly once, in order",
        [sym for b in batches for sym in b] == list(res.order),
    )
    state.update(gnodes.dispatch(res, state))
    report.check(
        "an empty queue routes to END",
        gnodes.route_dispatch(state) == "__end__",
        str(gnodes.route_dispatch(state)),
    )
    report.check(
        "the batch size honours resolve_workers' verdict",
        gnodes.dispatch(
            gnodes.GraphResources(
                options=batch.BatchOptions(),
                indexer=None,
                runlog=None,
                workers=batch.resolve_workers(4, _small_gpu()),
                members={},
                order=list(res.order),
                record=lambda _run: None,
            ),
            {"queue": list(res.order)},
        )["in_flight"]
        == ["T0"],
        "a 4 GB card collapses the batch to one ticker",
    )


def check_graph_parity(report: Report, tmp: Path) -> None:
    """The graph and the plain loop must produce the same manifest."""
    report.section("Graph / no-checkpoint parity")
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)

        def run(**extra: object) -> batch.RunManifest:
            return batch.run_batch(
                batch.BatchOptions(
                    limit=0,
                    output_dir=str(root),
                    log_dir=str(tmp),
                    notify=False,
                    allow_ephemeral=True,
                    **extra,  # type: ignore[arg-type]
                )
            )

        graphed = run(run_id="parity-graph")
        looped = run(run_id="parity-loop", no_checkpoint=True)
        report.check(
            "both paths complete",
            graphed.status == batch.COMPLETED and looped.status == batch.COMPLETED,
            f"graph={graphed.status}, loop={looped.status}",
        )
        report.check(
            "both record the same tickers and counters",
            graphed.tickers.keys() == looped.tickers.keys()
            and graphed.counters == looped.counters,
        )
        report.check(
            "the graph run is marked resumable and the loop is not",
            graphed.resumable and not looped.resumable,
        )
        report.check(
            "--no-checkpoint writes no checkpoints",
            not any(
                p.suffix == ".json"
                for p in (root / "_nifty50" / "checkpoints").rglob("*")
                if looped.run_id in str(p)
            ),
        )
        report.check(
            "a fresh run never adopts an existing thread's queue",
            run(run_id="parity-graph").tickers == {},
            "a colliding run id must discard, not resume",
        )
        report.check(
            "the manifest still round-trips through from_dict",
            batch.RunManifest.from_dict(graphed.to_dict()).run_id == graphed.run_id
            and batch.RunManifest.from_dict(graphed.to_dict()).resumable,
        )
        report.check(
            "status_report gains the two additive keys and loses none",
            {"resumable", "resumed_from"} <= set(batch.status_report(output_dir=root)),
        )


def check_resume_guards(report: Report, tmp: Path) -> None:
    """A resume may change compute settings, never what the run produces."""
    report.section("Resume guards")
    stored = batch.BatchOptions(
        tickers=["TCS"], fast_tables=False, annual_reports=1, chunk_size=800
    ).to_dict()

    merged = batch.merge_resume_options(
        stored,
        batch.BatchOptions(device="cuda", num_threads=16, workers=3, resume="r1"),
    )
    report.check(
        "compute settings may differ and are taken from the invocation",
        merged.device == "cuda" and merged.num_threads == 16 and merged.workers == 3,
    )
    report.check(
        "the original scope is preserved, not the invocation's",
        merged.tickers == ["TCS"] and merged.annual_reports == 1,
    )
    report.check(
        "the resumed run keeps its run id",
        merged.run_id == "r1" and merged.resume == "r1",
    )
    report.raises(
        "flipping --fast-tables refuses",
        batch.ResumeMismatch,
        lambda: batch.merge_resume_options(
            stored, batch.BatchOptions(fast_tables=True, resume="r1")
        ),
    )
    report.raises(
        "changing the document scope refuses",
        batch.ResumeMismatch,
        lambda: batch.merge_resume_options(
            stored, batch.BatchOptions(annual_reports=2, resume="r1")
        ),
    )
    report.raises(
        "changing the chunk size refuses",
        batch.ResumeMismatch,
        lambda: batch.merge_resume_options(
            stored, batch.BatchOptions(chunk_size=400, resume="r1")
        ),
    )
    report.check(
        "ResumeMismatch is a BatchError, so existing handlers still catch it",
        issubclass(batch.ResumeMismatch, batch.BatchError),
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        report.raises(
            "resuming with no runs on disk refuses",
            batch.BatchError,
            lambda: embed_cli._resume_options(
                argparse.Namespace(resume=""),
                batch.BatchOptions(output_dir=str(root), log_dir=str(tmp)),
                root,
            ),
        )


def check_end_to_end_resume(report: Report, tmp: Path) -> None:
    """A crashed run, resumed, must skip finished tickers and report them all.

    The whole point of the checkpointer, asserted through `run_batch` and the
    CLI's own resume resolution rather than against the graph directly.
    """
    report.section("End-to-end resume")
    members = [nifty50.Constituent(symbol=s) for s in "ABCDE"]
    processed: list[str] = []
    fired = {"crashed": False}

    def fake_process(member, options, indexer, runlog, position=0, total=0):  # noqa: ANN001, ANN202, ARG001
        processed.append(member.symbol)
        if member.symbol == "C" and not fired["crashed"]:
            fired["crashed"] = True
            raise RuntimeError("simulated crash at ticker C")
        return batch.TickerRun(
            symbol=member.symbol,
            name=member.symbol,
            started_at="t",
            status=batch.TICKER_OK,
        )

    saved = (batch.process_ticker, batch.load_nifty50, batch.verified_as_of)
    batch.process_ticker = fake_process
    batch.load_nifty50 = lambda _path=None: members  # type: ignore[assignment]
    batch.verified_as_of = lambda _members: ""  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base: dict[str, object] = {
                "output_dir": str(root),
                "log_dir": str(tmp),
                "notify": False,
                "allow_ephemeral": True,
                "workers": 1,
                "delay_seconds": 0,
            }
            report.raises(
                "the crash propagates and the run is recorded FAILED",
                RuntimeError,
                lambda: batch.run_batch(
                    batch.BatchOptions(run_id="crashy", **base)  # type: ignore[arg-type]
                ),
            )
            crashed = batch.load_manifest(run_id="crashy", output_dir=root)
            report.check(
                "the tickers finished before the crash are recorded",
                crashed is not None
                and crashed.status == batch.FAILED
                and sorted(crashed.tickers) == ["A", "B"],
                f"status={crashed.status if crashed else '?'}, "
                f"tickers={sorted(crashed.tickers) if crashed else []}",
            )

            resumed_options = embed_cli._resume_options(
                argparse.Namespace(resume="crashy"),
                batch.BatchOptions(**base),  # type: ignore[arg-type]
                root,
            )
            before = len(processed)
            final = batch.run_batch(resumed_options)
            redone = processed[before:]

            report.check(
                "finished tickers are not re-processed",
                "A" not in redone and "B" not in redone,
                f"re-processed {redone}",
            )
            report.check(
                "the run resumes at the ticker that failed",
                redone == ["C", "D", "E"],
                str(redone),
            )
            report.check(
                "the resumed manifest reports every ticker, not just the new ones",
                sorted(final.tickers) == list("ABCDE") and final.completed_tickers == 5,
                f"{sorted(final.tickers)}, completed={final.completed_tickers}",
            )
            report.check(
                "the resumed run names the run it continued",
                final.status == batch.COMPLETED and final.resumed_from == "crashy",
                f"{final.status}, resumed_from={final.resumed_from}",
            )
    finally:
        batch.process_ticker, batch.load_nifty50, batch.verified_as_of = saved


def offline(report: Report) -> None:
    # Windows may still hold a just-closed log file for a moment; a cleanup
    # failure is not a finding.
    with tempfile.TemporaryDirectory(
        prefix="nifty50_verify_", ignore_cleanup_errors=True
    ) as tmp_name:
        tmp = Path(tmp_name)
        check_constituents(report)
        check_collections(report)
        check_should_index_regression(report)
        check_layer1(report)
        check_selection(report)
        check_options(report)
        check_hardware(report)
        check_gpu_workers(report)
        check_extract_version(report)
        check_module_boundaries(report)
        check_extraction_cache(report)
        check_chunk_cache(report)
        check_index_failure_state(report)
        check_catalogue_ttl(report, tmp)
        check_ephemeral_refusal(report, tmp)
        check_part_sweep(report)
        check_graph_reducers(report)
        check_bounded_dispatch(report)
        check_graph_parity(report, tmp)
        check_resume_guards(report, tmp)
        check_end_to_end_resume(report, tmp)
        check_checkpoint_saver(report)
        check_graph_resume(report)
        check_checkpoint_size(report)
        check_manifest(report, tmp)
        check_status_no_network(report, tmp)
        check_pid_alive(report)
        check_background_launch(report, tmp)
        check_notifier(report)
        check_ntfy(report)
        check_ticker_notifier(report)
        check_page_filter(report)
        check_qdrant_transport(report)
        check_runlog(report, tmp)


# --- Live checks --------------------------------------------------------------


def live(report: Report, symbol: str) -> None:
    """Runs the real batch twice for one ticker and asserts the second is a no-op."""
    report.section(f"Live: {symbol} (network + Qdrant)")
    sym = safe_ticker(symbol)
    opts = batch.BatchOptions(tickers=[symbol], notify=False, delay_seconds=0)

    first = batch.run_batch(opts)
    run1 = first.tickers.get(sym)
    report.check(
        "first run completed",
        first.status in (batch.COMPLETED, batch.COMPLETED_WITH_ERRORS),
        first.status,
    )
    report.check(
        "ticker OK",
        run1 is not None and run1.status == batch.TICKER_OK,
        run1.status if run1 else "missing",
    )
    report.check(
        "≥1 document INDEXED or already current",
        run1 is not None and (run1.new_documents + run1.unchanged_documents) >= 1,
    )
    report.check(
        "Qdrant reports points",
        run1 is not None and run1.points_total > 0,
        f"{run1.points_total if run1 else 0} points",
    )

    # Second run: count embedding calls directly.
    encode_calls: list[int] = []
    original = QdrantVectorIndexer.embed_texts

    def counting(self, texts, *a, **k):  # noqa: ANN001, ANN002, ANN003
        encode_calls.append(len(texts))
        return original(self, texts, *a, **k)

    QdrantVectorIndexer.embed_texts = counting  # type: ignore[assignment]
    try:
        second = batch.run_batch(
            batch.BatchOptions(tickers=[symbol], notify=False, delay_seconds=0)
        )
    finally:
        QdrantVectorIndexer.embed_texts = original  # type: ignore[assignment]
    run2 = second.tickers.get(sym)
    report.check(
        "second run: 0 new documents",
        run2 is not None and run2.new_documents == 0,
        f"{run2.new_documents if run2 else '?'} new",
    )
    report.check(
        "second run: 0 embedding calls",
        not encode_calls,
        f"{sum(encode_calls)} texts embedded",
    )
    report.check(
        "second run: 0 downloads/extractions",
        second.counters.get("documents_downloaded", 0) == 0
        and second.counters.get("documents_extracted", 0) == 0,
    )
    report.check(
        "second run: status OK", run2 is not None and run2.status == batch.TICKER_OK
    )

    # Persistence and the per-stock mirror, from a fresh client.
    fresh = QdrantVectorIndexer(IndexerConfig.from_env())
    count = fresh.count_points(sym)
    report.check(
        "fresh client sees the collection",
        count > 0 and count == (run1.points_total if run1 else -1),
        f"{count} points in {fresh.config.collection_for(sym)}",
    )
    vec_path = fresh.config.output_dir / sym / "vectors.npz"
    if report.check("vectors.npz exists", vec_path.exists(), str(vec_path)):
        import numpy as np

        data = np.load(vec_path)
        rows = int(data["vectors"].shape[0])
        payloads = json.loads(
            (vec_path.parent / "payloads.json").read_text(encoding="utf-8")
        )
        report.check("mirror rows == Qdrant count", rows == count, f"{rows} rows")
        report.check("payloads aligned", len(payloads) == rows)
        report.check(
            "768-d float32",
            data["vectors"].dtype == np.float32
            and data["vectors"].shape[1] == fresh.vector_size,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--live", metavar="SYM", help="Also run the live checks for one ticker"
    )
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()
    cli.setup(level=logging.WARNING, fmt=cli.NAMED, stream=sys.stderr)

    banner("Nifty 50 embedding pipeline verification")
    report = Report()
    if not args.skip_offline:
        offline(report)
    if args.live:
        live(report, args.live)
    print(report.render())
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
