"""Locates the financial section of an annual report, before Docling sees it.

An Indian annual report is mostly not financial. Of Adani Enterprises' 396-page
FY2026 filing, the balance sheet, profit and loss, cash flow, notes, auditor's
report and MD&A occupy pages 222 onward; the 221 before them are the notice of
AGM, the directors' report and its annexures, the corporate governance report
and the business responsibility statement. Docling costs seconds per page, so
converting those 221 pages is the single largest avoidable expense in a run.

The trick that makes this cheap and safe: the financial content of an annual
report is a **contiguous tail**. Statements, notes and MD&A appear together, at
the back, and nothing financial follows them. So this module does not have to
classify every page -- it only has to find where that tail *starts*, which a
plain text pass over the PDF does in tens of milliseconds per page against
Docling's seconds. Docling then converts one contiguous `page_range`.

**Two rules this module exists to enforce.**

*Fail toward keeping pages.* Every uncertain outcome -- no anchor found, an
unreadable PDF, a document too short to have sections -- returns None, meaning
"convert the whole thing". Dropping a page makes it permanently unsearchable;
converting a spare one costs seconds. The asymmetry is not close.

*An anchor must be the financial statements, not a phrase near them.* The
motivating false positive is real: page 191 of that filing begins "INDEPENDENT
AUDITOR'S **CERTIFICATE** ON COMPLIANCE WITH THE CORPORATE GOVERNANCE
REQUIREMENTS". A pattern matching merely "independent auditor" anchors there
and drags in 31 pages of the governance report this module was written to
exclude. `AUDITOR_REPORT` therefore requires the word "report".

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["PAGE_FILTER_SUFFIX", "financial_page_range", "page_texts"]

# Appended to the extractor version when a run filters pages, because a
# filtered extraction is not the same document as an unfiltered one and Layer 1
# must re-extract rather than reuse it. Mirrors `FAST_TABLES_SUFFIX`.
PAGE_FILTER_SUFFIX: str = "+fin-pages"

# Characters of each page inspected. Section headings sit at the top of a page;
# reading further finds the phrase "balance sheet" in running prose and anchors
# on a page that merely mentions one.
HEAD_CHARS: int = 900

# A document shorter than this has no section structure worth filtering -- a
# concall transcript or an investor deck -- so it is always converted whole.
MIN_PAGES: int = 60

# Anchors for the start of the financial tail. Each must name a statement, the
# notes, or the auditor's *report*; see the module docstring on why "auditor"
# alone is not allowed.
ANCHORS: tuple[str, ...] = (
    r"management\s+discussion\s+and\s+analysis",
    # "AUDITOR'S REPORT", "AUDITORS' REPORT", curly or straight apostrophe.
    r"independent\s+auditor.{0,3}\s+report",
    r"balance\s+sheet",
    r"statement\s+of\s+profit\s+and\s+loss",
    r"cash\s+flow\s+statement",
    r"statement\s+of\s+cash\s+flow",
    r"statement\s+of\s+changes\s+in\s+equity",
    r"notes?\s+(?:to|forming\s+part\s+of)\s+the",
    r"significant\s+accounting\s+policies",
    r"(?:standalone|consolidated)\s+financial\s+statements?",
)

_ANCHOR_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in ANCHORS
)


def page_texts(pdf: Path | str) -> list[str]:
    """Extracts each page's raw text, cheaply and without raising.

    Args:
        pdf: The PDF to read.

    Returns:
        One string per page, empty where a page yields nothing. An empty list
        when the file cannot be opened at all -- the caller reads that as
        "no opinion", not "no pages".
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is a Docling dependency
        logger.warning("pypdf unavailable; page filtering disabled for this run.")
        return []

    try:
        reader = PdfReader(str(pdf))
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF is Docling's problem
        logger.warning("Could not open %s for page filtering: %s", pdf, exc)
        return []

    out: list[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            out.append("")
    return out


def financial_page_range(
    pdf: Path | str, min_pages: int = MIN_PAGES
) -> tuple[int, int] | None:
    """Returns the 1-based inclusive page range holding the financial content.

    Args:
        pdf: The annual report to inspect.
        min_pages: Documents shorter than this are never filtered.

    Returns:
        ``(start, end)`` for Docling's `page_range`, or None to convert the
        whole document. None is returned whenever the answer is uncertain --
        see the module docstring on failing toward keeping pages.
    """
    texts = page_texts(pdf)
    total = len(texts)
    if total < min_pages:
        return None

    readable = sum(1 for t in texts if len(t.strip()) >= 20)
    if readable < total // 2:
        # A scanned or image-only filing: the text pass cannot see headings, so
        # any range it proposed would be a guess over missing evidence.
        logger.info(
            "Page filter: only %d of %d pages have extractable text; "
            "converting the whole document.",
            readable,
            total,
        )
        return None

    for index, text in enumerate(texts):
        head = text[:HEAD_CHARS]
        for pattern in _ANCHOR_RE:
            if pattern.search(head):
                start = index + 1
                if start == 1:
                    # The first page anchored, so there is nothing to drop --
                    # usually a contents page listing the statements.
                    return None
                logger.info(
                    "Page filter: financial section starts at page %d of %d "
                    "(%r); dropping %d page(s).",
                    start,
                    total,
                    pattern.pattern[:40],
                    start - 1,
                )
                return start, total

    logger.info(
        "Page filter: no financial-section anchor in %d pages; "
        "converting the whole document.",
        total,
    )
    return None
