"""Report orchestration: snapshot to compiled PDF.

Charts and the generated Typst source are written into the stock's own
directory, because Typst resolves `#image` paths relative to the source file.
Once the PDF compiles they are swept away: they are reproducible from the
cached payloads, and what a reader wants in that directory is the report.
"""

from datetime import date
import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple

import typst

from reporting import analytics
from reporting import charts as charts_module
from reporting import composites as composites_module
from reporting import selfcheck
from reporting import typst_doc
from core.config import OUTPUT_DIR, safe_ticker
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
BUILD_GLOBS: Tuple[str, ...] = ("*.svg", "*.typ")
PREVIEW_DIR: str = "preview"


class ReportError(RuntimeError):
    """Raised when a report cannot be produced."""


def generate_report(
    ticker: str,
    output_dir: Optional[Path] = None,
    refresh: bool = False,
    as_of: Optional[str] = None,
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

    logger.info("[%s] fetching collector data", symbol)
    payloads = CollectorClient().fetch_all(symbol, refresh=refresh)
    snapshot = build_snapshot(symbol, payloads)

    if not snapshot.years and not snapshot.quarters:
        raise ReportError(
            "No income statement data for %s; refusing to render an empty report" % symbol
        )
    for warning in snapshot.warnings:
        logger.warning("[%s] %s", symbol, warning)

    logger.info("[%s] computing derived analytics", symbol)
    derived = analytics.compute(snapshot)
    for note in derived.notes:
        logger.info("[%s] %s", symbol, note)

    logger.info("[%s] computing composites", symbol)
    composites = composites_module.compute(snapshot, derived)
    for note in composites.notes:
        logger.info("[%s] %s", symbol, note)

    # Verification runs before rendering, not after, so that a broken
    # identity is on the page rather than only in a log nobody reads.
    check = selfcheck.run(snapshot, derived, composites)
    for failure in check.failures:
        logger.error("[%s] self-check failed: %s (residual %s %s)",
                     symbol, failure.name, failure.delta, failure.unit)

    logger.info("[%s] rendering charts", symbol)
    produced = charts_module.render_all(snapshot, derived, composites, work_dir)

    # Load qualitative research findings if generated
    findings_path = work_dir / "findings" / "findings.json"
    findings_data = None
    if findings_path.exists():
        try:
            import json
            findings_data = json.loads(findings_path.read_text(encoding="utf-8"))
            logger.info("[%s] loaded qualitative research findings from %s", symbol, findings_path)
        except Exception as exc:
            logger.warning("[%s] could not load qualitative findings: %s", symbol, exc)

    stamp = as_of or date.today().strftime("%d %b %Y")
    source = typst_doc.build_document(
        snapshot, derived, composites, check, produced, as_of=stamp, findings=findings_data
    )

    source_path = work_dir / (folded + "_report.typ")
    source_path.write_text(source, encoding="utf-8")

    logger.info("[%s] compiling %d chars of Typst source", symbol, len(source))
    try:
        pdf_bytes = typst.compile(str(source_path))
    except Exception as exc:
        raise ReportError(
            "Typst compilation failed for %s. Source retained at %s.\n%s"
            % (symbol, source_path, exc)
        ) from exc

    pdf_path = work_dir / (folded + "_report.pdf")
    pdf_path.write_bytes(pdf_bytes)
    logger.info("[%s] wrote %s (%.1f KB)", symbol, pdf_path, pdf_path.stat().st_size / 1024)

    if not keep_build:
        swept = _sweep_build(work_dir, pdf_path)
        logger.info("[%s] swept %d build artefact(s); the PDF is what remains",
                    symbol, swept)
    return pdf_path


def _sweep_build(work_dir: Path, pdf_path: Path) -> int:
    """Removes the build byproducts from a stock's directory.

    Args:
        work_dir: The stock's directory.
        pdf_path: The report, which is never removed.

    Returns:
        How many entries were removed.
    """
    removed = 0
    for pattern in BUILD_GLOBS:
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
