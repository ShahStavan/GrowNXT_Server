"""Report orchestration: snapshot to compiled PDF.

Charts, the generated Typst source and the output PDF all land in one
directory per ticker. Typst resolves `#image` paths relative to the source
file, so co-locating them is what lets the document reference charts by
bare filename.
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
from reporting.client import CollectorClient
from reporting.snapshot import CompanySnapshot, build_snapshot

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "output"


class ReportError(RuntimeError):
    """Raised when a report cannot be produced."""


def generate_report(
    ticker: str,
    output_dir: Optional[Path] = None,
    refresh: bool = False,
    as_of: Optional[str] = None,
    keep_source: bool = True,
) -> Path:
    """Builds the institutional PDF report for one ticker.

    Args:
        ticker: Stock ticker symbol, e.g. 'WIPRO'.
        output_dir: Root output directory. Defaults to `output/`.
        refresh: Re-request collector data instead of using the cache.
        as_of: Display date for the header and disclaimer. Defaults to today.
        keep_source: Retain the generated `.typ` alongside the PDF, which
            makes a layout problem inspectable after the fact.

    Returns:
        Path to the written PDF.

    Raises:
        ReportError: If the snapshot is too sparse to report on, or Typst
            fails to compile the generated source.
    """
    symbol = ticker.upper().strip()
    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    work_dir = root / symbol
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

    source_path = work_dir / (symbol + "_report.typ")
    source_path.write_text(source, encoding="utf-8")

    logger.info("[%s] compiling %d chars of Typst source", symbol, len(source))
    try:
        pdf_bytes = typst.compile(str(source_path))
    except Exception as exc:
        raise ReportError(
            "Typst compilation failed for %s. Source retained at %s.\n%s"
            % (symbol, source_path, exc)
        ) from exc

    pdf_path = work_dir / (symbol + "_report.pdf")
    pdf_path.write_bytes(pdf_bytes)

    if not keep_source:
        source_path.unlink(missing_ok=True)

    logger.info("[%s] wrote %s (%.1f KB)", symbol, pdf_path, pdf_path.stat().st_size / 1024)
    return pdf_path


def snapshot_for(ticker: str, refresh: bool = False) -> CompanySnapshot:
    """Builds a snapshot without rendering, for inspection and tests."""
    symbol = ticker.upper().strip()
    return build_snapshot(symbol, CollectorClient().fetch_all(symbol, refresh=refresh))
