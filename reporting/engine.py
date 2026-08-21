"""Report orchestration: snapshot to compiled PDF.

Charts and the generated Typst source land in one workspace per ticker,
because Typst resolves `#image` paths relative to the source file and chart
file names repeat across tickers. The compiled PDF is written to the shared
`reports/` directory instead, so every stock's finished report sits together.
"""

from datetime import date
import logging
from pathlib import Path
from typing import Optional

import typst

from reporting import analytics
from reporting import charts as charts_module
from reporting import composites as composites_module
from reporting import selfcheck
from reporting import typst_doc
from core.config import OUTPUT_DIR, REPORTS_DIR, safe_ticker
from reporting.client import CollectorClient
from reporting.snapshot import build_snapshot

logger = logging.getLogger(__name__)

# The API server, the CLI and the ingestion layer all write into the same
# per-stock directory, so the roots and the folding rule are owned by config.
DEFAULT_OUTPUT_DIR: Path = OUTPUT_DIR
DEFAULT_REPORTS_DIR: Path = REPORTS_DIR


class ReportError(RuntimeError):
    """Raised when a report cannot be produced."""


def generate_report(
    ticker: str,
    output_dir: Optional[Path] = None,
    refresh: bool = False,
    as_of: Optional[str] = None,
    keep_source: bool = True,
    reports_dir: Optional[Path] = None,
) -> Path:
    """Builds the institutional PDF report for one ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'WIPRO'.
        output_dir: Root of the per-ticker build workspace, holding the charts
            and the Typst source. Defaults to `output/`.
        refresh: Re-request collector data instead of using the cache.
        as_of: Display date for the header and disclaimer. Defaults to today.
        keep_source: Retain the generated `.typ` in the workspace, which makes
            a layout problem inspectable after the fact.
        reports_dir: Directory collecting every stock's finished PDF. Defaults
            to `reports/`.

    Returns:
        Path to the written PDF, inside `reports_dir`.

    Raises:
        ReportError: If the snapshot is too sparse to report on, or Typst
            fails to compile the generated source.
    """
    # The unfolded symbol addresses the collector; the folded one addresses
    # the filesystem, because Indian symbols carry ampersands.
    symbol = ticker.upper().strip()
    folded = safe_ticker(symbol)
    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    reports_root = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
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

    stamp = as_of or date.today().strftime("%d %b %Y")
    source = typst_doc.build_document(
        snapshot, derived, composites, check, produced, as_of=stamp)

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

    reports_root.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_root / (folded + "_report.pdf")
    pdf_path.write_bytes(pdf_bytes)

    if not keep_source:
        source_path.unlink(missing_ok=True)

    logger.info("[%s] wrote %s (%.1f KB)", symbol, pdf_path, pdf_path.stat().st_size / 1024)
    return pdf_path
