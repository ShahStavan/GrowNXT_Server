"""Locates an annual report's financial section, before Docling sees it.

The financial content is a contiguous tail, so this only finds where that tail
starts -- a pypdf pass at ~60 ms/page against Docling's seconds. Every
uncertain case returns None, meaning convert everything. Both rules and the
ADANIENT measurements: `.claude/specs/document-acquisition.md` section 5.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["PAGE_FILTER_SUFFIX", "financial_page_range", "page_texts"]

# Part of the extractor version: a filtered extraction is not the same
# document as a full one and must not satisfy its cache entry.
PAGE_FILTER_SUFFIX: str = "+fin-pages"

# Headings sit at the top of a page; reading further matches "balance sheet"
# in running prose and anchors on a page that merely mentions one.
HEAD_CHARS: int = 900

# Below this there is no section structure to filter -- a transcript or deck.
MIN_PAGES: int = 60

ANCHORS: tuple[str, ...] = (
    r"management\s+discussion\s+and\s+analysis",
    # "report" is required: a bare "independent auditor" also matches page
    # 191's governance CERTIFICATE and drags in 31 pages. Guarded by
    # check_page_filter.
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

    An empty list means the file could not be opened at all -- read that as
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

    None means convert the whole document, and is returned for every
    uncertain case.
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
