"""Embed the Nifty 50 corpus into Qdrant, incrementally.

    python -m scripts.embed_nifty50                      # all 50, incremental
    python -m scripts.embed_nifty50 --tickers TCS INFY   # just these
    python -m scripts.embed_nifty50 --force TCS          # re-process TCS's known filings
    python -m scripts.embed_nifty50 --force-all          # re-process every ticker in scope
    python -m scripts.embed_nifty50 --limit 3 --dry-run  # catalogue + diff only
    python -m scripts.embed_nifty50 --background         # detach; returns a run id
    python -m scripts.embed_nifty50 --status [--json]    # latest run, no network
    python -m scripts.embed_nifty50 --status --run-id 20260903T041500Z --tail 20
    python -m scripts.embed_nifty50 --hardware           # resolved device/threads
    python -m scripts.embed_nifty50 --device cuda --threads 4 --fast-tables

Each ticker gets its newest annual report, concall transcript and investor
presentation (one of each by default), embedded into its own Qdrant
collection and mirrored to ``output/<TICKER>/vectors.npz``. A second run
against unchanged filings does one catalogue request per ticker and nothing
else. Logs land under ``logs/nifty50/`` -- see `ingestion.runlog`.

Exit codes: 0 every ticker OK; 1 at least one ticker PARTIAL/FAILED; 2 the
run could not start (already running, bad arguments).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv  # noqa: E402

# Loaded here, at the entry point, rather than relying on `ingestion.indexer`
# importing dotenv as a side effect. Both the Qdrant URL and the ntfy webhook
# live in .env, and the failure mode of not finding them is quiet in exactly
# the wrong way: a detached run would embed into an in-memory store and never
# say a word about it. `--hardware` and `--status` reach neither import.
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from core.config import safe_ticker  # noqa: E402
from ingestion import batch  # noqa: E402
from ingestion.batch import BatchOptions, RunManifest, TickerRun  # noqa: E402
from scripts import cli  # noqa: E402

logger = logging.getLogger("scripts.embed_nifty50")

EXIT_OK: int = 0
EXIT_ERRORS: int = 1
EXIT_REFUSED: int = 2

# Docling's table post-processor logs a WARNING per recovered cell -- hundreds
# per annual report. They belong in the run log, not on the operator's console.
CONSOLE_NOISE: frozenset[str] = frozenset(
    {"MatchingPostProcessor", "docling", "httpx", "urllib3", "sentence_transformers"}
)


class _ConsoleNoiseFilter(logging.Filter):
    """Drops sub-ERROR chatter from third-party internals on the console only."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        top = record.name.split(".")[0]
        return top not in CONSOLE_NOISE


def _grid(rows: list[list[str]], header: list[str]) -> str:
    """Renders rows as a fixed-width text table."""
    table = [header] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(header))]
    lines = [
        " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in table
    ]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def print_ticker(run: TickerRun, manifest: RunManifest) -> None:
    """Prints one ticker's document table as it completes."""
    print(
        f"\n[{manifest.completed_tickers}/{manifest.total_tickers}] {run.symbol}"
        f"{' - ' + run.name if run.name else ''}: {run.status}"
        f" ({run.new_documents} new, {run.unchanged_documents} unchanged,"
        f" {run.failed_documents} failed, {run.points_total} points, {run.elapsed_seconds:.1f}s)"
    )
    if run.documents:
        rows = [
            [
                d.doc_id,
                d.status,
                d.reason or "-",
                str(d.chunks),
                f"{d.seconds:.1f}s",
                (d.error[:60] if d.error else ""),
            ]
            for d in sorted(run.documents, key=lambda d: d.doc_id)
        ]
        print(_grid(rows, ["doc_id", "status", "reason", "chunks", "time", "error"]))
    for err in run.errors:
        if not err.get("doc_id"):
            print(f"  ! {err.get('stage')}: {err.get('error')}")


def print_summary(manifest: RunManifest) -> None:
    """Prints the closing per-ticker table and the one-line verdict."""
    rows = [
        [
            sym,
            str(run.new_documents),
            str(run.unchanged_documents),
            str(run.failed_documents),
            str(run.points_total),
            run.status,
        ]
        for sym, run in sorted(manifest.tickers.items())
    ]
    print(
        "\n" + _grid(rows, ["ticker", "new", "unchanged", "failed", "points", "status"])
    )
    c = manifest.counters
    print(
        f"\n{c.get('tickers_ok', 0)}/{manifest.total_tickers} tickers OK, "
        f"{c.get('tickers_partial', 0)} partial, {c.get('tickers_failed', 0)} failed, "
        f"{c.get('tickers_no_documents', 0)} without documents."
    )
    print(
        f"Documents: {c.get('documents_catalogued', 0)} catalogued, "
        f"{c.get('documents_selected', 0)} selected, {c.get('documents_pending', 0)} pending, "
        f"{c.get('documents_downloaded', 0)} downloaded, "
        f"{c.get('documents_reused_from_cache', 0)} reused, "
        f"{c.get('documents_extracted', 0)} extracted, {c.get('documents_chunked', 0)} chunked, "
        f"{c.get('documents_embedded', 0)} embedded ({c.get('chunks_embedded', 0)} chunks)."
    )
    hw = manifest.hardware
    if hw:
        print(
            f"Compute: {hw.get('device', '?')}"
            f"{' (' + hw['gpu_name'] + ')' if hw.get('gpu_name') else ''}, "
            f"{hw.get('num_threads', '?')} thread(s), "
            f"embed batch {hw.get('embed_batch_size', '?')}"
            f"{', fp16' if hw.get('embed_fp16') else ''}."
        )
    print(f"Run {manifest.run_id}: {manifest.status}; logs in {manifest.log_dir}")


def print_status(data: dict, tail: int) -> None:
    """Prints a status report in text form."""
    status = data.get("status")
    if status in ("NO_RUNS", "UNKNOWN_RUN"):
        print(f"{status}. Known runs: {', '.join(data.get('runs') or []) or 'none'}")
        return
    print(
        f"run {data.get('run_id')}: {status}"
        f"{' (pid alive)' if data.get('pid_alive') else ''} - "
        f"{data.get('completed_tickers', 0)}/{data.get('total_tickers', 0)} tickers "
        f"({data.get('progress', 0) * 100:.0f}%), started {data.get('started_at')}"
        f"{', finished ' + data['finished_at'] if data.get('finished_at') else ''}"
    )
    print(
        f"model={data.get('embedding_model')} collections={data.get('collection_mode')} "
        f"qdrant={data.get('qdrant_target')} logs={data.get('log_dir')}"
    )
    tickers = data.get("tickers") or {}
    if tickers:
        rows = [
            [
                sym,
                str(run.get("new_documents", 0)),
                str(run.get("unchanged_documents", 0)),
                str(run.get("failed_documents", 0)),
                str(run.get("points_total", 0)),
                str(run.get("status", "")),
            ]
            for sym, run in sorted(tickers.items())
        ]
        print(_grid(rows, ["ticker", "new", "unchanged", "failed", "points", "status"]))
    counters = data.get("counters") or {}
    if counters:
        print("counters: " + ", ".join(f"{k}={v}" for k, v in counters.items()))
    if data.get("error"):
        print(f"error: {data['error']}")
    if tail > 0 and data.get("log_dir"):
        log = Path(data["log_dir"]) / "run.log"
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"\n--- last {min(tail, len(lines))} lines of {log} ---")
            print("\n".join(lines[-tail:]))


def print_hardware(options: BatchOptions, as_json: bool = False) -> None:
    """Reports the compute a run with these options would resolve to.

    Answers the question that decides whether a fifty-document backfill takes
    a night or a week -- which device the models will actually land on -- and
    answers it before anything is downloaded.
    """
    from core.hardware import profile

    resolved = profile(
        device=options.device,
        num_threads=options.num_threads,
        embed_batch_size=options.embed_batch_size,
    )
    if as_json:
        data = resolved.to_dict()
        data["workers"] = batch.resolve_workers(options.workers, resolved)
        data["idle_gpu"] = resolved.idle_gpu
        print(json.dumps(data, indent=2))
        return

    rows = [
        ["device", resolved.device],
        ["accelerator", resolved.gpu_name or "-"],
        ["vram", f"{resolved.vram_gb:g} GB" if resolved.vram_gb else "-"],
        ["cpu cores (physical)", str(resolved.cpu_cores)],
        ["ram", f"{resolved.ram_gb:g} GB" if resolved.ram_gb else "-"],
        ["extraction threads", str(resolved.num_threads)],
        ["embed batch size", str(resolved.embed_batch_size)],
        ["embed precision", "float16" if resolved.embed_fp16 else "float32"],
        ["tables", "fast" if options.fast_tables else "accurate"],
        ["pages", "financial only" if options.page_filter else "all"],
        ["workers", str(batch.resolve_workers(options.workers, resolved))],
        ["torch", resolved.torch_build or "not installed"],
        ["torch cuda build", resolved.cuda_build or "none (CPU wheel)"],
        ["overridden", ", ".join(resolved.overrides) or "-"],
    ]
    print(_grid(rows, ["setting", "value"]))
    if resolved.idle_gpu:
        print(
            "\nWARNING: this machine has an NVIDIA GPU that the installed torch "
            "cannot use.\n         Every model will run on the CPU. Install a "
            "CUDA build of torch to\n         use the card -- see README, "
            "'GPU acceleration'."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed Nifty 50 filings into Qdrant, incrementally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = parser.add_argument_group("scope")
    scope.add_argument(
        "--tickers", nargs="+", metavar="SYM", help="Restrict to these symbols"
    )
    scope.add_argument(
        "--nifty50-file", metavar="PATH", help="Override ticker_mapping.csv"
    )
    scope.add_argument("--limit", type=int, metavar="N", help="First N tickers only")
    scope.add_argument(
        "--annual-reports",
        type=int,
        default=1,
        metavar="N",
        help="Newest annual reports per ticker (default 1)",
    )
    scope.add_argument(
        "--transcripts",
        type=int,
        default=1,
        metavar="N",
        help="Newest concall transcripts per ticker (default 1)",
    )
    scope.add_argument(
        "--presentations",
        type=int,
        default=1,
        metavar="N",
        help="Newest investor presentations per ticker (default 1)",
    )
    scope.add_argument(
        "--concall-years",
        type=int,
        default=1,
        metavar="N",
        help="Years of concalls to request from the catalogue (default 1)",
    )

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--force",
        nargs="*",
        metavar="SYM",
        help="Re-process known filings for these symbols (no value: all)",
    )
    behaviour.add_argument(
        "--force-all", action="store_true", help="Re-process every ticker in scope"
    )
    behaviour.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"Tickers in parallel (1-{batch.MAX_WORKERS})",
    )
    behaviour.add_argument(
        "--dry-run", action="store_true", help="Catalogue and diff only"
    )
    behaviour.add_argument(
        "--delay",
        type=float,
        default=batch.DEFAULT_INTER_TICKER_DELAY,
        help="Seconds between tickers",
    )
    behaviour.add_argument(
        "--catalog-ttl",
        type=float,
        metavar="HOURS",
        help="Skip the catalogue request for a ticker whose filings are all "
        f"embedded and whose last check is this recent (default "
        f"{batch.DEFAULT_CATALOG_TTL_HOURS:g}, or ${batch.CATALOG_TTL_ENV}; "
        "0 disables)",
    )
    behaviour.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run the plain sequential loop with no checkpointing (the "
        "pre-graph behaviour)",
    )
    behaviour.add_argument(
        "--allow-ephemeral",
        action="store_true",
        help="Run even though Qdrant is in-memory. Vectors are lost on exit "
        "while state.json records them as INDEXED, so the next run skips them",
    )
    behaviour.add_argument(
        "--run-id", metavar="ID", help="Explicit run id (default: timestamp)"
    )
    behaviour.add_argument(
        "--output-dir", metavar="PATH", help="Override GROWNXT_OUTPUT_DIR"
    )
    behaviour.add_argument("--log-dir", metavar="PATH", help="Override GROWNXT_LOG_DIR")
    behaviour.add_argument(
        "--notify-webhook", metavar="URL", help="Override NIFTY50_NOTIFY_WEBHOOK_URL"
    )
    behaviour.add_argument(
        "--no-notify", action="store_true", help="Suppress the webhook for this run"
    )
    behaviour.add_argument(
        "--no-ticker-notify",
        action="store_true",
        help="Only notify at the end of the run, not as each ticker finishes",
    )
    behaviour.add_argument("--verbose", action="store_true", help="DEBUG logging")

    compute = parser.add_argument_group("compute")
    compute.add_argument(
        "--device",
        metavar="DEV",
        help="auto|cuda|cuda:N|cpu|mps|xpu for Docling and the embedder "
        "(default: auto-detect)",
    )
    compute.add_argument(
        "--threads",
        type=int,
        metavar="N",
        help="CPU threads for extraction (default: physical cores)",
    )
    compute.add_argument(
        "--embed-batch-size",
        type=int,
        metavar="N",
        help="Texts per encode call (default: sized from VRAM)",
    )
    compute.add_argument(
        "--fast-tables",
        action="store_true",
        help="TableFormer fast mode: quicker, and re-extracts every document "
        "previously indexed in accurate mode",
    )
    compute.add_argument(
        "--page-filter",
        action="store_true",
        help="Convert only an annual report's financial section (statements, "
        "notes, auditor's report, MD&A), skipping the AGM notice, directors' "
        "report and governance sections. Re-extracts anything indexed without "
        "it. Short documents and unrecognised layouts are converted whole",
    )

    modes = parser.add_argument_group("modes")
    modes.add_argument(
        "--background",
        "--detach",
        dest="background",
        action="store_true",
        help="Launch detached and return immediately",
    )
    modes.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="RUN_ID",
        help="Continue a run from its checkpoint (no value: the newest "
        "unfinished run). Prefer this to --force after a crash",
    )
    modes.add_argument(
        "--forget-run",
        metavar="RUN_ID",
        help="Delete a run's checkpoints. Never touches state.json or vectors",
    )
    modes.add_argument(
        "--status",
        action="store_true",
        help="Report the latest (or --run-id) run; no network",
    )
    modes.add_argument(
        "--hardware",
        action="store_true",
        help="Report the resolved device, threads and batch size, then exit",
    )
    modes.add_argument("--json", action="store_true", help="Machine-readable output")
    modes.add_argument(
        "--pending-only",
        action="store_true",
        help="With --status: print just the symbols still to do, ready to "
        "paste after --tickers",
    )
    modes.add_argument(
        "--tail",
        type=int,
        default=0,
        metavar="N",
        help="With --status: last N log lines",
    )
    modes.add_argument("--options-file", metavar="PATH", help=argparse.SUPPRESS)
    return parser


def options_from_args(args: argparse.Namespace) -> BatchOptions:
    if args.options_file:
        data = json.loads(Path(args.options_file).read_text(encoding="utf-8"))
        return BatchOptions.from_dict(data)
    force_all = bool(args.force_all) or (
        args.force is not None and len(args.force) == 0
    )
    return BatchOptions(
        tickers=list(args.tickers or []),
        nifty50_file=args.nifty50_file,
        annual_reports=max(0, args.annual_reports),
        transcripts=max(0, args.transcripts),
        presentations=max(0, args.presentations),
        concall_years=max(1, args.concall_years),
        force=list(args.force or []),
        force_all=force_all,
        limit=args.limit,
        workers=args.workers,
        dry_run=bool(args.dry_run),
        run_id=args.run_id,
        notify=not args.no_notify,
        notify_per_ticker=not args.no_ticker_notify,
        webhook_url=args.notify_webhook,
        verbose=bool(args.verbose),
        delay_seconds=max(0.0, args.delay),
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        num_threads=args.threads,
        embed_batch_size=args.embed_batch_size,
        fast_tables=bool(args.fast_tables),
        page_filter=bool(args.page_filter),
        catalog_ttl_hours=_catalog_ttl(args.catalog_ttl),
        allow_ephemeral=bool(args.allow_ephemeral),
        no_checkpoint=bool(args.no_checkpoint),
    )


def _catalog_ttl(requested: float | None) -> float:
    """Resolves the catalogue TTL from the flag, the environment, or the default."""
    if requested is not None:
        return max(0.0, float(requested))
    raw = os.getenv(batch.CATALOG_TTL_ENV, "").strip()
    if not raw:
        return batch.DEFAULT_CATALOG_TTL_HOURS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %g hours.",
            batch.CATALOG_TTL_ENV,
            raw,
            batch.DEFAULT_CATALOG_TTL_HOURS,
        )
        return batch.DEFAULT_CATALOG_TTL_HOURS


def pending_symbols(data: dict, options: BatchOptions) -> list[str]:
    """Returns the symbols a run still has to do, in constituent order.

    The manifest alone is not enough: a run that died at ticker forty has **no
    entry at all** for the ten it never reached, so selecting on recorded
    status silently omits exactly the tickers the operator most needs. The
    scope is therefore taken from the constituent list and the finished ones
    subtracted.

    Args:
        data: A `batch.status_report` payload.
        options: The invocation's options, for the ticker scope.

    Returns:
        Directory-safe symbols still to process.
    """
    recorded = data.get("tickers") or {}
    finished = {
        sym
        for sym, run in recorded.items()
        if run.get("status") in (batch.TICKER_OK, batch.TICKER_NO_DOCUMENTS)
    }
    try:
        scope = [safe_ticker(m.symbol) for m in options.resolve_constituents()]
    except Exception as exc:  # noqa: BLE001 - fall back to what the manifest knows
        logger.warning("could not load the constituent list: %s", exc)
        scope = sorted(recorded)
    return [sym for sym in scope if sym not in finished]


def _resume_options(
    args: argparse.Namespace, requested: BatchOptions, output_dir: Path
) -> BatchOptions:
    """Resolves which run to resume and the options to resume it under.

    Args:
        args: The parsed invocation.
        requested: The options this invocation was given.
        output_dir: The artefact root, for finding the manifest.

    Returns:
        The merged options, carrying the original run's scope.

    Raises:
        BatchError: If no run can be resumed, or a locked setting differs.
    """
    run_id = (args.resume or "").strip() or _newest_unfinished(output_dir)
    if not run_id:
        raise batch.BatchError(
            "no unfinished run to resume; known runs: "
            + (", ".join(batch.list_run_ids(output_dir)) or "none")
        )
    manifest = batch.load_manifest(run_id=run_id, output_dir=output_dir)
    if manifest is None:
        raise batch.BatchError(f"no manifest recorded for run {run_id}")
    if manifest.status in (batch.COMPLETED, batch.COMPLETED_WITH_ERRORS):
        # Its dispatch queue was fully consumed, so a resume would find nothing
        # to do and say nothing about it. Failed *documents* are retried by an
        # ordinary run: Layer 1 re-queues anything state.json does not record
        # as INDEXED.
        raise batch.BatchError(
            f"run {run_id} already reached every ticker ({manifest.status}); "
            "resuming it would do nothing. Re-run normally to retry the "
            "documents that failed."
        )
    if manifest.status == batch.RUNNING and batch.pid_alive(manifest.pid):
        raise batch.BatchAlreadyRunning(
            f"run {run_id} is still RUNNING (pid {manifest.pid}); "
            "not resuming a live run"
        )
    requested.resume = run_id
    return batch.merge_resume_options(manifest.options, requested)


def _newest_unfinished(output_dir: Path) -> str | None:
    """Returns the newest run that did not reach a terminal success, if any."""
    for run_id in batch.list_run_ids(output_dir):
        manifest = batch.load_manifest(run_id=run_id, output_dir=output_dir)
        if manifest is None:
            continue
        # Only a run that stopped before working through its queue has
        # anything to resume. One that reached every ticker but recorded
        # failures is retried by an ordinary run, not by a resume.
        if manifest.status in (batch.FAILED, batch.RUNNING):
            return run_id
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli.setup(
        level=logging.DEBUG if args.verbose else logging.INFO,
        fmt=cli.TIMED,
        datefmt=cli.CLOCK,
        stream=sys.stderr,
    )
    # The console is for the operator; module chatter still reaches the run log.
    if not args.verbose:
        for handler in logging.getLogger().handlers:
            handler.addFilter(_ConsoleNoiseFilter())

    options = options_from_args(args)
    output_dir = options.output_root

    if args.hardware:
        print_hardware(options, as_json=args.json)
        return EXIT_OK

    if args.forget_run:
        from ingestion.graph import JsonFileSaver

        JsonFileSaver(output_dir=output_dir).delete_thread(args.forget_run)
        print(f"forgot checkpoints for run {args.forget_run}")
        return EXIT_OK

    if args.resume is not None:
        try:
            options = _resume_options(args, options, output_dir)
        except batch.BatchError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        print(f"resuming run {options.run_id}")

    if args.status:
        data = batch.status_report(run_id=args.run_id, output_dir=output_dir)
        if args.pending_only:
            symbols = pending_symbols(data, options)
            print(json.dumps(symbols) if args.json else " ".join(symbols))
            return EXIT_OK
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print_status(data, tail=args.tail)
        return EXIT_OK

    if args.background:
        try:
            run_id, log_path, pid = batch.launch_background(options)
        except batch.BatchAlreadyRunning as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "pid": pid,
                        "log": str(log_path),
                        "status": "RUNNING",
                    }
                )
            )
        else:
            print(f"started run {run_id} (pid {pid})")
            print(f"  log:    {log_path}")
            print(
                f"  status: python -m scripts.embed_nifty50 --status --run-id {run_id}"
            )
        return EXIT_OK

    try:
        manifest = batch.run_batch(
            options, on_ticker=None if args.json else print_ticker
        )
    except batch.BatchAlreadyRunning as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except batch.BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_summary(manifest)
    bad = {batch.TICKER_FAILED, batch.TICKER_PARTIAL}
    return (
        EXIT_ERRORS
        if any(r.status in bad for r in manifest.tickers.values())
        else EXIT_OK
    )


if __name__ == "__main__":
    sys.exit(main())
