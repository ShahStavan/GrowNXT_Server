"""Report orchestration: snapshot to compiled PDF.

Charts and the generated Typst source are written into the stock's own
directory, because Typst resolves `#image` paths relative to the source file.
Once the PDF compiles they are swept away: they are reproducible from the
cached payloads, and what a reader wants in that directory is the report.
"""

import logging
import shutil
from datetime import date
from pathlib import Path

import typst

from core.config import OUTPUT_DIR, safe_ticker
from reporting import (
    analytics,
    charts as charts_module,
    composites as composites_module,
    selfcheck,
    typst_doc,
)
from reporting.client import CollectorClient
from reporting.snapshot import build_snapshot

logger = logging.getLogger(__name__)

# The API server, the CLI and the ingestion layer all write into the same
# per-stock directory, so the root and the folding rule are owned by config.
DEFAULT_OUTPUT_DIR: Path = OUTPUT_DIR

# Byproducts of a build: the chart SVGs the document referenced, the generated
# Typst source, and any rendered page previews. Once the PDF exists these are
# reproducible from the cached payloads, so they are swept away.
#
# Only these. The same directory holds the ingestion layer's caches and the
# Drive upload record, which belong to other layers: clearing those would
# re-download and re-embed a company's filings, and re-upload a report that
# had not changed.
BUILD_GLOBS: tuple[str, ...] = ("*.svg", "*.typ")
PREVIEW_DIR: str = "preview"


class ReportError(RuntimeError):
    """Raised when a report cannot be produced."""


def generate_report(
    ticker: str,
    output_dir: Path | None = None,
    refresh: bool = False,
    as_of: str | None = None,
    keep_build: bool = False,
) -> Path:
    """Builds the institutional PDF report for one ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'WIPRO'.
        output_dir: Root holding one directory per stock. Defaults to
            `output/`.
        refresh: Re-request collector data instead of using the cache.
        as_of: Display date for the header and disclaimer. Defaults to today.
        keep_build: Retain the chart SVGs and the generated `.typ` instead of
            sweeping them, which makes a layout problem inspectable after the
            fact. A failed compile keeps them regardless.

    Returns:
        Path to the written PDF, in the stock's own directory.

    Raises:
        ReportError: If the snapshot is too sparse to report on, or Typst
            fails to compile the generated source.
    """
    # The unfolded symbol addresses the collector; the folded one addresses
    # the filesystem, because Indian symbols carry ampersands.
    symbol = ticker.upper().strip()
    folded = safe_ticker(symbol)
    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    work_dir = root / folded
    work_dir.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    # Define Track 1: Quantitative statements, ratios, composites, self-check & charts
    def _run_quantitative_track():
        logger.info(
            "[%s] [Track 1] Fetching collector data & computing analytics...", symbol
        )
        client = CollectorClient(cache_dir=root)
        payloads = client.fetch_all(symbol, refresh=refresh)
        snap = build_snapshot(symbol, payloads)

        if not snap.years and not snap.quarters:
            raise ReportError(
                f"No income statement data for {symbol}; refusing to render an empty report"
            )
        for warning in snap.warnings:
            logger.warning("[%s] %s", symbol, warning)

        der = analytics.compute(snap)
        for note in der.notes:
            logger.info("[%s] %s", symbol, note)

        comp = composites_module.compute(snap, der)
        for note in comp.notes:
            logger.info("[%s] %s", symbol, note)

        chk = selfcheck.run(snap, der, comp)
        for failure in chk.failures:
            logger.error(
                "[%s] self-check failed: %s (residual %s %s)",
                symbol,
                failure.name,
                failure.delta,
                failure.unit,
            )

        logger.info("[%s] [Track 1] Rendering charts...", symbol)
        prod = charts_module.render_all(snap, der, comp, work_dir)
        return snap, der, comp, chk, prod

    # Define Track 2: Qualitative RAG research synthesis
    def _run_qualitative_track():
        findings_path = work_dir / "findings" / "findings.json"
        findings_data = None

        if not findings_path.exists() or refresh:
            try:
                logger.info(
                    "[%s] [Track 2] Running RAG pipeline to generate institutional findings...",
                    symbol,
                )
                from ingestion.rag.pipeline import extract_ticker_findings

                dossier = extract_ticker_findings(
                    ticker=symbol,
                    company_name=symbol,
                    force=refresh,
                )
                findings_data = dossier.to_dict()
                logger.info(
                    "[%s] [Track 2] Qualitative RAG synthesis complete (%d pillars).",
                    symbol,
                    len(dossier.pillars),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] RAG qualitative extraction skipped: %s", symbol, exc
                )

        if findings_path.exists() and not findings_data:
            try:
                import json

                findings_data = json.loads(findings_path.read_text(encoding="utf-8"))
                logger.info(
                    "[%s] Loaded qualitative research findings from %s",
                    symbol,
                    findings_path,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Could not load qualitative findings: %s", symbol, exc
                )

        return findings_data

    # Execute Track 1 and Track 2 concurrently
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"report-{symbol}"
    ) as executor:
        f_quant = executor.submit(_run_quantitative_track)
        f_qual = executor.submit(_run_qualitative_track)
        snapshot, derived, composites, check, produced = f_quant.result()
        findings_data = f_qual.result()

    stamp = as_of or date.today().strftime("%d %b %Y")
    source = typst_doc.build_document(
        snapshot,
        derived,
        composites,
        check,
        produced,
        as_of=stamp,
        findings=findings_data,
    )

    source_path = work_dir / (folded + "_report.typ")
    source_path.write_text(source, encoding="utf-8")

    logger.info("[%s] compiling %d chars of Typst source", symbol, len(source))
    try:
        pdf_bytes = typst.compile(str(source_path))
    except Exception as exc:
        raise ReportError(
            f"Typst compilation failed for {symbol}. Source retained at "
            f"{source_path}.\n{exc}"
        ) from exc

    pdf_path = work_dir / (folded + "_report.pdf")
    pdf_path.write_bytes(pdf_bytes)
    logger.info(
        "[%s] wrote %s (%.1f KB)", symbol, pdf_path, pdf_path.stat().st_size / 1024
    )

    # Automatically upload / mirror report to Google Drive if credentials exist
    try:
        from storage.gdrive import credentials_present, ensure_uploaded

        if credentials_present():
            logger.info("[%s] uploading / updating report on Google Drive...", symbol)
            drive_file = ensure_uploaded(pdf_path, symbol, force=refresh)
            logger.info(
                "[%s] Google Drive update complete: %s", symbol, drive_file.view_link
            )
    except Exception as exc:
        logger.warning("[%s] Google Drive update skipped: %s", symbol, exc)

    if not keep_build:
        swept = _sweep_build(work_dir, pdf_path)
        logger.info(
            "[%s] swept %d build artefact(s); the PDF and findings are what remain",
            symbol,
            swept,
        )
    return pdf_path


def _sweep_build(work_dir: Path, pdf_path: Path) -> int:
    """Removes the build byproducts and temporary files from a stock's directory.

    Args:
        work_dir: The stock's directory.
        pdf_path: The report, which is never removed.

    Returns:
        How many entries were removed.
    """
    removed = 0
    patterns = list(BUILD_GLOBS) + ["*.tmp", "*.temp"]
    for pattern in patterns:
        for path in work_dir.glob(pattern):
            if path == pdf_path:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:  # pragma: no cover - a locked viewer, say
                logger.warning("could not remove %s: %s", path, exc)

    preview = work_dir / PREVIEW_DIR
    if preview.is_dir():
        shutil.rmtree(preview, ignore_errors=True)
        removed += 1
    return removed
