"""Layout-aware text extraction for filing PDFs.

Annual reports and investor presentations are typeset in two or three columns,
and several Indian issuers publish A3 spreads that hold two logical pages side
by side. A naive ``page.extract_text()`` walks such a page line by line across
the full width, so the left column's text is spliced into the right column's
mid-sentence:

    26) As the 80th AGM is being held through VC, the route experience in the
    map is not annexed to this Notice. operations of the Company over a long

Chunks built from that are unusable as retrieval evidence. This module runs a
recursive XY-cut instead: it finds the vertical whitespace gutters of a block,
assigns each line to a column, and emits the columns in reading order,
recursing into each one. Full-width elements (section headings, table rows) are
horizontal cuts that flush the columns above them, but only across the columns
they actually cover -- on a spread, the left page's tables must not interrupt
the right page's prose. The recursion is what makes a spread work at all: at
the top level the only gutter wide enough to survive is the one between the two
logical pages, and every threshold has to be re-measured against one page's
width before that page's own columns become visible.

Three further filing-specific quirks are handled here:

* Rotated sidebar furniture. Issuers print the report title vertically in the
  page margin, and pdfplumber extracts it as reversed nonsense -- a title set
  bottom-to-top arrives one word at a time, spelled backwards. Every such word
  is non-upright, so a single ``upright`` filter removes the whole class.
* Folio numbers. A bare page number parked in the outer margin stretches the
  page's text block by 40 points, and the empty band between it and the body
  then reads as a gutter -- which on a spread swallows the real one.
* pdfminer's colour-space complaints. Malformed separation colour spaces are
  harmless for text extraction but flood stderr, so the pdfminer loggers are
  quietened on import.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

logger = logging.getLogger(__name__)

# pdfminer reports every malformed colour operator it meets. A long annual
# report can emit thousands of these; none of them affect extracted text.
for _noisy in ("pdfminer", "pdfminer.pdfinterp", "pdfminer.converter",
               "pdfminer.pdfdocument", "pdfminer.pdfpage", "pdfplumber"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared in requirements
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False


# --- Tunables -----------------------------------------------------------------

# Vertical whitespace band that counts as a column gutter, in PDF points.
MIN_GUTTER_WIDTH: float = 8.0

# Fraction of a block's rows allowed to print at a position that is otherwise
# gutter-like. Set by the full-width table rows of a financial statement, which
# must not be able to hide the gutter between the prose columns beside them.
GUTTER_INK_TOLERANCE: float = 0.18

# Rows required before gutters are looked for at all.
MIN_GUTTER_ROWS: int = 5

# Smallest column, as a fraction of the block width, a gutter may leave behind.
MIN_COLUMN_RATIO: float = 0.14

# Slack, in points, between a line fragment's start and its column's left edge.
COLUMN_ALIGN_TOLERANCE: float = 4.5

# Whitespace, in points, that sets a table cell apart from the cell before it.
# Prose puts a single word space around its figures; a table column does not.
TABLE_CELL_GAP: float = 6.0

# Whitespace, in points, that isolates a bare number enough to be a folio.
FOLIO_ISOLATION: float = 18.0

# Words whose baselines sit within this many points belong to the same line.
LINE_Y_TOLERANCE: float = 3.0

# Guard against pathological gutter detection on sparse or graphical pages.
MAX_COLUMNS: int = 6

# Recursion limit for the XY-cut. Two levels cover an A3 spread of two-column
# pages; deeper cuts start slicing table cells apart.
MAX_CUT_DEPTH: int = 2

# Private-use glyphs are bullet and icon fonts with no Unicode meaning.
_PRIVATE_USE_RE = re.compile("[-]")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RUN_RE = re.compile("[ \t  -​]{2,}")
# A table cell: an optionally signed, bracketed or suffixed number.
_NUMERIC_CELL_RE = re.compile(r"^[(\[]?[-+]?[₹$]?[\d][\d,.]*%?[)\]]?[*]?$")
_SOFT_SPACE_RE = re.compile("[   ​]")

_DASH_MAP = {
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": " - ",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "•": "* ",
}


def normalize_text(text: str) -> str:
    """Normalizes extracted PDF text without destroying numeric formatting.

    Ligatures and full-width digits are folded to their ASCII equivalents,
    private-use glyphs (bullet fonts) become spaces, and runs of spaces
    collapse. Currency symbols, thousands separators, and parenthesised
    negatives are left intact because downstream figures depend on them.

    Args:
        text: Raw text as returned by the extractor.

    Returns:
        Normalized text with whitespace stripped from each line.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _PRIVATE_USE_RE.sub(" ", out)
    out = _CONTROL_RE.sub(" ", out)
    out = _SOFT_SPACE_RE.sub(" ", out)
    for source, target in _DASH_MAP.items():
        out = out.replace(source, target)
    lines = [_SPACE_RUN_RE.sub(" ", line).strip() for line in out.split("\n")]
    return "\n".join(lines)


@dataclass
class PageText:
    """One extracted page.

    Attributes:
        number: 1-based page number in the PDF's page order.
        text: Reading-order text for the page.
        lines: Individual text lines in reading order.
        width: Page width in points.
        height: Page height in points.
        n_columns: Number of detected columns (1 for single-column pages).
        n_words: Count of upright words used.
        is_spread: True when the page is wide enough to be a two-page spread.
        cut_depth: Deepest XY-cut level used to order the page.
        header: Text printed in the page's top band, used for section
            classification of annual reports.
    """

    number: int
    text: str = ""
    lines: List[str] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    n_columns: int = 1
    n_words: int = 0
    is_spread: bool = False
    cut_depth: int = 0
    header: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serialisable view used by the parse cache."""
        return {
            "number": self.number,
            "text": self.text,
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "n_columns": self.n_columns,
            "n_words": self.n_words,
            "is_spread": self.is_spread,
            "cut_depth": self.cut_depth,
            "header": self.header,
        }


def _group_words_into_lines(
    words: Sequence[Dict[str, Any]],
    y_tolerance: float = LINE_Y_TOLERANCE,
) -> List[List[Dict[str, Any]]]:
    """Groups words into visual lines by vertical position.

    Args:
        words: pdfplumber word dictionaries.
        y_tolerance: Maximum baseline difference within one line, in points.

    Returns:
        Lines, each a list of words sorted left to right.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [ordered[0]]
    anchor = float(ordered[0]["top"])
    for word in ordered[1:]:
        top = float(word["top"])
        if abs(top - anchor) <= y_tolerance:
            current.append(word)
        else:
            lines.append(sorted(current, key=lambda w: float(w["x0"])))
            current = [word]
            anchor = top
    lines.append(sorted(current, key=lambda w: float(w["x0"])))
    return lines


def _line_span(line: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """Returns the horizontal extent of a line as (x0, x1)."""
    return float(line[0]["x0"]), max(float(w["x1"]) for w in line)


def _is_tabular_row(row: Sequence[Dict[str, Any]]) -> bool:
    """Returns True when a row reads as a row of a numeric table.

    Table rows must never be cut at a column gutter: a statement of cash flows
    printed beside two prose columns puts its labels on one side of the gutter
    and its figures on the other, and splitting there files the figures away
    from the line item they belong to. Two or more numeric cells separated by
    tabular whitespace is the signature to protect.
    """
    isolated = 0
    previous = float(row[0]["x1"])
    for word in row[1:]:
        gap = float(word["x0"]) - previous
        if gap >= TABLE_CELL_GAP and _NUMERIC_CELL_RE.match(str(word["text"])):
            isolated += 1
        previous = max(previous, float(word["x1"]))
    return isolated >= 2


def _split_row_at_gutters(
    row: Sequence[Dict[str, Any]],
    gutters: Sequence[Tuple[float, float]],
    alignment_tolerance: float = COLUMN_ALIGN_TOLERANCE,
) -> List[List[Dict[str, Any]]]:
    """Splits one visual row wherever it crosses a detected column gutter.

    Words at the same height in two different columns form a single visual row,
    so a row is not a unit of reading order. Splitting at the page's gutters --
    and only there -- recovers the per-column line fragments while leaving
    intact the wide inter-word gaps inside a full-width line.

    A gap over the gutter is necessary but not sufficient. Justified prose
    stretches its word spaces, and a full-width justified line can open a
    gutter-sized gap at exactly the gutter's position; cutting there would file
    the tail of the sentence into the next column and emit it out of order. The
    word after a real gutter starts at its column's left edge, so requiring
    that alignment separates the two cases.

    Args:
        row: Words of one visual row, sorted left to right.
        gutters: Gutter bands detected for the block.
        alignment_tolerance: Points of slack allowed between a fragment's start
            and its column's left edge.

    Returns:
        Fragments in left-to-right order.
    """
    if not gutters or _is_tabular_row(row):
        return [list(row)]

    fragments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_x1: Optional[float] = None
    for word in row:
        x0 = float(word["x0"])
        crossed = previous_x1 is not None and any(
            previous_x1 <= end and x0 >= start and x0 <= end + alignment_tolerance
            for start, end in gutters
        )
        if crossed:
            fragments.append(current)
            current = []
        current.append(word)
        previous_x1 = float(word["x1"]) if previous_x1 is None else max(previous_x1, float(word["x1"]))
    if current:
        fragments.append(current)
    return fragments


def _content_bounds(
    words: Sequence[Dict[str, Any]],
    quantile: float = 0.02,
) -> Tuple[float, float]:
    """Returns the horizontal bounds of a page's text block.

    Plain minima and maxima are unusable here: a single folio number parked in
    the outer margin widens the block by 40 points, and the empty band between
    it and the body then reads as a column gutter -- which on an A3 spread
    swallows the real gutter and re-interleaves the columns. Trimmed quantiles
    ignore that handful of margin words while tracking the body text exactly.

    Args:
        words: Upright words of the page.
        quantile: Fraction of words trimmed from each edge.

    Returns:
        The (x0, x1) bounds of the text block.
    """
    lefts = sorted(float(w["x0"]) for w in words)
    rights = sorted(float(w["x1"]) for w in words)
    index = min(len(lefts) - 1, int(quantile * len(lefts)))
    return lefts[index], rights[len(rights) - 1 - index]


def find_gutters(
    rows: Sequence[Sequence[Dict[str, Any]]],
    content_x0: float,
    content_x1: float,
    min_gap: float = MIN_GUTTER_WIDTH,
    ink_tolerance: float = GUTTER_INK_TOLERANCE,
    min_rows: int = MIN_GUTTER_ROWS,
) -> List[Tuple[float, float]]:
    """Finds the vertical whitespace gutters that separate columns.

    A position is gutter-like when almost no row prints anything at it. The
    tolerance is what makes this work on filings: a strict "no row prints here"
    scan is defeated by the one centred heading or the handful of full-width
    table rows that cross the gutter, while a per-row majority vote is defeated
    from the other side, because columns typeset with different leading share
    no baselines, so *no* row crosses the gutter for a majority to be drawn
    from. Counting rows with ink at each position, and allowing a minority of
    them, survives both.

    Args:
        rows: Visual rows of the block, each a list of words.
        content_x0: Left edge of the block's text.
        content_x1: Right edge of the block's text.
        min_gap: Minimum band width, in points, to count as a gutter.
        ink_tolerance: Fraction of rows allowed to print at a gutter position.
        min_rows: Minimum rows required before gutters are looked for at all.

    Returns:
        Gutter bands as (start, end) pairs ordered left to right, each strictly
        inside the content block and wide enough to leave usable columns.
    """
    content_width = content_x1 - content_x0
    if content_width <= 0 or len(rows) < min_rows:
        return []

    origin = int(content_x0)
    size = max(1, int(content_width) + 2)

    # Difference array: one increment pair per word, then a prefix sum. A page
    # costs O(words + width) instead of O(rows * width).
    ink = [0] * (size + 2)
    for row in rows:
        for word in row:
            start = min(max(0, int(float(word["x0"])) - origin), size)
            end = min(max(0, int(float(word["x1"])) + 1 - origin), size)
            if end > start:
                ink[start] += 1
                ink[end] -= 1

    max_ink = ink_tolerance * len(rows)
    gutters: List[Tuple[float, float]] = []
    run_start: Optional[int] = None
    total = 0
    for index in range(size):
        total += ink[index]
        if total <= max_ink:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= min_gap:
                gutters.append((float(origin + run_start), float(origin + index)))
            run_start = None
    if run_start is not None and size - run_start >= min_gap:
        gutters.append((float(origin + run_start), float(origin + size)))

    # Bands touching an edge are the block's own margin, not a gutter.
    gutters = [g for g in gutters if g[0] > content_x0 + 1 and g[1] < content_x1 - 1]

    # A gutter that leaves a sliver of a column is the ragged right edge of a
    # single-column page, not a real division.
    minimum = MIN_COLUMN_RATIO * content_width
    kept: List[Tuple[float, float]] = []
    cursor = content_x0
    for start, end in gutters:
        if (start - cursor) < minimum or (content_x1 - end) < minimum:
            continue
        kept.append((start, end))
        cursor = end
    gutters = kept

    if len(gutters) + 1 > MAX_COLUMNS:
        # Sparse graphical pages produce many spurious bands; keep the widest.
        gutters = sorted(gutters, key=lambda g: g[1] - g[0], reverse=True)[: MAX_COLUMNS - 1]
        gutters.sort()
    return gutters


def _columns_from_gutters(
    gutters: Sequence[Tuple[float, float]],
    content_x0: float,
    content_x1: float,
) -> List[Tuple[float, float]]:
    """Converts gutter bands into column bands."""
    bounds: List[Tuple[float, float]] = []
    cursor = content_x0
    for start, end in gutters:
        bounds.append((cursor, start))
        cursor = end
    bounds.append((cursor, content_x1))
    return [b for b in bounds if b[1] > b[0]]


def _assign_column(
    line: Sequence[Dict[str, Any]],
    columns: Sequence[Tuple[float, float]],
) -> int:
    """Returns the index of the column a line belongs to, by its midpoint."""
    x0, x1 = _line_span(line)
    midpoint = (x0 + x1) / 2.0
    for index, (start, end) in enumerate(columns):
        if start <= midpoint <= end:
            return index
    # Fall back to the nearest column when the midpoint lands inside a gutter.
    distances = [min(abs(midpoint - start), abs(midpoint - end)) for start, end in columns]
    return distances.index(min(distances))


def _render_line(line: Sequence[Dict[str, Any]]) -> str:
    """Joins a line's words into a single string."""
    return " ".join(str(word["text"]) for word in line).strip()


def _spanned_columns(
    fragment: Sequence[Dict[str, Any]],
    columns: Sequence[Tuple[float, float]],
) -> List[int]:
    """Returns the indices of the columns a fragment has ink in.

    More than one index means the fragment bridges a gutter, which makes it a
    full-width element across exactly those columns.
    """
    x0, x1 = _line_span(fragment)
    return [
        index for index, (start, end) in enumerate(columns)
        if x1 > start + COLUMN_ALIGN_TOLERANCE and x0 < end - COLUMN_ALIGN_TOLERANCE
    ]


def _strip_folio_numbers(
    row: Sequence[Dict[str, Any]],
    page_height: float,
) -> List[Dict[str, Any]]:
    """Drops printed page numbers from a row in the top or bottom margin.

    Folio numbers are not part of the narrative, and left in place they both
    split sentences during chunking and stretch the page's text block far
    enough to fake a column gutter. Isolation is what identifies them: a bare
    number set well away from its neighbours is a folio, whereas the year in a
    dateline such as "July 16, 2026" sits a word space from the day and must
    survive. Section running heads are kept deliberately -- they are the
    strongest signal available for classifying an annual report's sections.

    Args:
        row: Words of one visual row, sorted left to right.
        page_height: Page height in points.

    Returns:
        The row without its folio numbers.
    """
    band = 0.09 * page_height
    top = float(row[0]["top"])
    if band < top < page_height - band:
        return list(row)

    kept: List[Dict[str, Any]] = []
    for index, word in enumerate(row):
        text = str(word["text"]).strip()
        if not text.isdigit() or len(text) > 4:
            kept.append(word)
            continue
        left_gap = (
            float(word["x0"]) - float(row[index - 1]["x1"]) if index else FOLIO_ISOLATION
        )
        right_gap = (
            float(row[index + 1]["x0"]) - float(word["x1"])
            if index + 1 < len(row) else FOLIO_ISOLATION
        )
        if left_gap < FOLIO_ISOLATION or right_gap < FOLIO_ISOLATION:
            kept.append(word)
    return kept


def _header_text(
    words: Sequence[Dict[str, Any]],
    page_height: float,
    band_ratio: float = 0.11,
) -> str:
    """Returns the text printed in the page's top band.

    Args:
        words: Body words of the page.
        page_height: Page height in points.
        band_ratio: Fraction of the page height treated as the header band.

    Returns:
        Space-joined header text, used by the annual-report section classifier.
    """
    limit = band_ratio * page_height
    band = [w for w in words if float(w["top"]) <= limit]
    if not band:
        return ""
    parts = [_render_line(row) for row in _group_words_into_lines(band)]
    return normalize_text(" ".join(p for p in parts if p)).replace("\n", " ").strip()


def _count_columns(
    words: Sequence[Dict[str, Any]],
    content_x0: float,
    content_x1: float,
) -> int:
    """Returns the number of top-level columns detected on a page."""
    rows = [
        row for row in _group_words_into_lines(words)
        if not _is_tabular_row(row)
    ]
    gutters = find_gutters(rows, content_x0, content_x1)
    return len(_columns_from_gutters(gutters, content_x0, content_x1)) or 1


def _cut(
    words: Sequence[Dict[str, Any]],
    x0: float,
    x1: float,
    depth: int = 0,
) -> Tuple[List[str], int]:
    """Orders a block of words by recursive XY-cut.

    One pass is not enough for an A3 spread. At the top level the only gutter
    wide enough to survive is the one between the two logical pages, and the
    full-width tables printed on the left page are narrow relative to the
    spread, so they are not recognised as horizontal cuts and their cells fill
    the gutter that separates the left page's own two text columns. Recursing
    into each column re-measures both thresholds against that column's width,
    which is what makes the nested case come out in reading order.

    Args:
        words: Words of the block.
        x0: Left bound of the block.
        x1: Right bound of the block.
        depth: Current recursion depth.

    Returns:
        A tuple of (ordered lines, maximum depth reached).
    """
    rows = _group_words_into_lines(words)
    # Table rows are excluded from the search: their cells sit wherever the
    # table's own columns fall, which is unrelated to the page's text columns.
    prose = [row for row in rows if not _is_tabular_row(row)]
    gutters = find_gutters(prose, x0, x1) if depth < MAX_CUT_DEPTH else []
    columns = _columns_from_gutters(gutters, x0, x1)

    if len(columns) <= 1:
        return [_render_line(row) for row in rows if _render_line(row)], depth

    buckets: List[List[Dict[str, Any]]] = [[] for _ in columns]
    ordered: List[str] = []
    reached = depth

    def flush(indices: Sequence[int]) -> None:
        """Recurses into the named buffered columns in order, then clears them."""
        nonlocal reached
        for index in indices:
            bucket = buckets[index]
            if not bucket:
                continue
            start, end = columns[index]
            lines, sub_depth = _cut(bucket, start, end, depth + 1)
            ordered.extend(lines)
            reached = max(reached, sub_depth)
            del bucket[:]

    every = list(range(len(columns)))
    for row in rows:
        for fragment in _split_row_at_gutters(row, gutters):
            rendered = _render_line(fragment)
            if not rendered:
                continue
            spanned = _spanned_columns(fragment, columns)
            if len(spanned) > 1:
                # A fragment bridging a gutter is a horizontal cut, but only
                # across the columns it actually covers. On an A3 spread the
                # left page's full-width tables must not interrupt the right
                # page's columns, which are a separate block of reading.
                flush(spanned)
                ordered.append(rendered)
                continue
            buckets[_assign_column(fragment, columns)].extend(fragment)
    flush(every)
    return ordered, reached


def extract_page(page: Any, number: int) -> PageText:
    """Extracts one page in reading order.

    Args:
        page: A ``pdfplumber.page.Page``.
        number: 1-based page number for the returned record.

    Returns:
        A populated PageText. Pages with no upright words yield empty text.
    """
    width = float(page.width)
    height = float(page.height)
    try:
        words = page.extract_words(extra_attrs=["upright"])
    except Exception as exc:  # pragma: no cover - malformed page objects
        logger.warning("Word extraction failed on page %d: %s", number, exc)
        return PageText(number=number, width=width, height=height)

    upright = [w for w in words if w.get("upright", True)]
    result = PageText(
        number=number,
        width=width,
        height=height,
        n_words=len(upright),
        # A portrait page is taller than wide; a landscape spread of two such
        # pages is at least twice as wide as it is tall in aspect terms.
        is_spread=width > 1.35 * height,
    )
    if not upright:
        return result

    body: List[Dict[str, Any]] = []
    for row in _group_words_into_lines(upright):
        body.extend(_strip_folio_numbers(row, height))
    if not body:
        return result

    content_x0, content_x1 = _content_bounds(body)
    ordered, depth = _cut(body, content_x0, content_x1)
    result.n_columns = _count_columns(body, content_x0, content_x1)
    result.cut_depth = depth
    result.header = _header_text(body, height)
    # Normalize once, at the line level, so that `lines` and `text` agree:
    # parsers work from `lines`, and glyph substitutions applied to only one of
    # the two would reach the chunks through whichever the parser happened to
    # read.
    normalized = normalize_text("\n".join(line for line in ordered if line))
    result.lines = [line for line in normalized.split("\n") if line]
    result.text = "\n".join(result.lines)
    return result


def extract_pdf_pages(
    path: Path,
    max_pages: Optional[int] = None,
    page_numbers: Optional[Iterable[int]] = None,
) -> List[PageText]:
    """Extracts every page of a PDF in reading order.

    Args:
        path: Path to the PDF on disk.
        max_pages: Optional cap on the number of pages read, from the front.
        page_numbers: Optional explicit 1-based page numbers to extract.

    Returns:
        Extracted pages in document order. An unreadable PDF yields whatever
        was extracted before the failure rather than raising, so one bad filing
        cannot abort an ingest.

    Raises:
        RuntimeError: If pdfplumber is not installed.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is required for document parsing; install it first.")

    pages: List[PageText] = []
    wanted = set(page_numbers) if page_numbers is not None else None
    try:
        with pdfplumber.open(str(path)) as pdf:
            total = len(pdf.pages)
            limit = total if max_pages is None else min(total, max_pages)
            for index in range(limit):
                number = index + 1
                if wanted is not None and number not in wanted:
                    continue
                page = pdf.pages[index]
                pages.append(extract_page(page, number))
                # pdfplumber caches per-page objects; releasing keeps an
                # annual report of several hundred pages inside a few hundred
                # megabytes.
                page.flush_cache()
    except Exception as exc:
        logger.error("Failed to extract %s: %s", path, exc)
        return pages
    return pages


def page_count(path: Path) -> int:
    """Returns a PDF's page count, or 0 when it cannot be read."""
    try:
        import pypdf
        with open(str(path), "rb") as handle:
            return len(pypdf.PdfReader(handle).pages)
    except Exception as exc:
        logger.warning("Could not read page count for %s: %s", path, exc)
        return 0
