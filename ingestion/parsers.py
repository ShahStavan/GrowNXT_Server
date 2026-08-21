"""Type-aware parsing of filings into sections and blocks.

The three document classes in the catalogue carry their content in entirely
different structures, and flattening them into one bag of pages throws away the
metadata that later makes retrieval precise:

* An **annual report** is a bound volume of unrelated documents -- an AGM
  notice, a board report, management's discussion, the risk section, the
  statutory financial statements, an ESG report. Retrieving "revenue growth
  commentary" from the AGM notice is a wrong answer that reads like a right
  one, so every page is classified into a section and the section travels with
  the chunk. Issuers print the section name as a running head on each page,
  which is a far stronger signal than any font-size heuristic.
* A **concall transcript** is a dialogue. Prepared remarks are management's
  own framing; the Q&A is where analysts press on what the remarks avoided.
  They have different evidentiary weight in a research report, so the
  transcript is split at the moderator's handover and every turn keeps its
  speaker and the speaker's role.
* An **investor presentation** is a deck of near-independent slides, where the
  slide title is the only context a figure has.

Nothing here is specific to one issuer. Headings an issuer builds from its own
name are derived from the catalogued company name at parse time, and a covering
letter's sign-off is matched by shape rather than against any list of company
names.

Parsing is separated from chunking on purpose: parse output is cached, and
re-chunking with different parameters must not require re-reading a filing of
several hundred pages.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ingestion.layout import PageText, extract_pdf_pages, normalize_text

logger = logging.getLogger(__name__)

PARSER_VERSION: str = "parser/v1"

# Version per document class. A change to the transcript parser must re-parse
# the transcripts and nothing else: re-embedding a several-hundred-page annual
# report costs minutes and a few hundred thousand tokens, and its parser did
# not change.
PARSER_VERSIONS: Dict[str, str] = {
    "annual_report": "annual/v5",
    "concall_transcript": "transcript/v2",
    "concall_presentation": "presentation/v1",
}


def parser_version(doc_type: str) -> str:
    """Returns the parser version for one document class."""
    return PARSER_VERSIONS.get(doc_type, PARSER_VERSION)


# --- Section vocabularies -----------------------------------------------------

# Canonical annual-report sections, with the phrases that identify them. Order
# matters only for tie-breaking; scoring decides.
ANNUAL_SECTIONS: List[Tuple[str, str, List[str]]] = [
    ("mdna", "Management Discussion and Analysis", [
        "management discussion", "management's discussion", "leadership insights",
        "results of operations", "liquidity and capital resources", "outlook",
        "financial performance", "operating results", "revenue by",
        # Indian issuers open the narrative with a signed letter, which is
        # management's own account of the year and belongs with the MD&A.
        "chairman's statement", "chairman's message", "chairmans message",
        "from the chairman", "letter to shareholders", "letter to the shareholders",
        "managing director's", "ceo's message", "message from the",
        "financial highlights", "ten year highlights",
        # "Decade at a Glance" is deliberately absent: it contains the business
        # section's "at a glance", both would score equally on the same heading,
        # and the tie would be broken arbitrarily. The shorter phrase claims it.
    ]),
    ("business", "Business and Segments", [
        "corporate overview", "about us", "about the company", "who we are",
        "at a glance", "our business", "business model", "business segment",
        "segment review", "global business lines", "strategic market units",
        "value creation", "our strategy", "reporting context",
        "products and platforms", "service lines", "industry sectors",
        "our products", "our brands", "manufacturing facilities", "our presence",
        "operational highlights", "business overview", "our portfolio",
    ]),
    ("risk", "Risk Management", [
        "risk management", "risk factors", "principal risks", "enterprise risk",
        "key risks", "risk governance", "internal financial control",
        "risks and concerns", "risk mitigation",
    ]),
    ("financials", "Financial Statements", [
        "financial statements", "balance sheet", "statement of profit and loss",
        "statement of cash flows", "notes to the", "independent auditor",
        "auditor's report", "significant accounting policies",
        "consolidated financial", "standalone financial", "schedules to",
        "cash flow statement", "value added statement",
        "statement of changes in equity", "notes forming part",
    ]),
    ("governance", "Corporate Governance", [
        "corporate governance", "board of directors", "board's report",
        "boards report", "directors' report", "director's report", "remuneration",
        "committee", "related party", "secretarial audit", "shareholding pattern",
        "general shareholder information", "statutory report", "annexure",
        "corporate information",
        # "Management Discussion and Analysis" is formally an annexure to the
        # Board's Report in an Indian filing, so listing it here is tempting.
        # It must not be: the phrase would score for governance as well as for
        # mdna, the tie would break arbitrarily, and any MD&A page that landed
        # in governance would then be dropped by the default skip list. Pages
        # of one issuer's MD&A were observed doing exactly that.
    ]),
    ("esg", "Sustainability and ESG", [
        "business responsibility", "brsr", "sustainability", "climate",
        "environmental", "social and relationship", "human capital",
        "corporate social responsibility", "csr", "diversity", "esg",
        "natural capital", "assurance statement", "intellectual capital",
        "manufactured capital", "social capital",
    ]),
    ("notice", "Notice of Annual General Meeting", [
        "notice to members", "annual general meeting", "e-voting", "evoting",
        "postal ballot", "proxy form", "explanatory statement", "agm",
        "notice of the", "attendance slip", "route map",
    ]),
]

# How far down a page to look for a section heading, and how long a line may be
# and still be one. The scan has to reach past the running head and any sidebar
# column the XY-cut emits first, without reading the whole page as headings.
HEADING_SCAN_LINES: int = 14
MAX_HEADING_WORDS: int = 12

# A page announcing this many different sections is an index, not a section.
MAX_ANNOUNCED_SECTIONS: int = 3

# Heading-like lines allowed to announce a section on one page.
ANNOUNCING_HEADINGS: int = 2

# Words dropped from a company's name before it is used as a section keyword.
# "About <name>" and "<name> at a glance" are headings; "About Limited" is not,
# and matching on a legal suffix alone would fire on any page that mentions one.
_COMPANY_SUFFIXES = {
    "limited", "ltd", "ltd.", "plc", "inc", "inc.", "corporation", "corp",
    "company", "co", "co.", "industries", "enterprises", "holdings", "group",
    "india", "and", "&", "the", "of",
}

# Headings an issuer builds from its own name.
_COMPANY_HEADING_TEMPLATES = [
    "about %s",
    "%s at a glance",
    "%s overview",
    "this is %s",
    "%s in brief",
]

SECTION_TITLES: Dict[str, str] = {key: title for key, title, _ in ANNUAL_SECTIONS}
SECTION_TITLES["other"] = "Other Disclosures"
SECTION_TITLES["prepared_remarks"] = "Prepared Remarks"
SECTION_TITLES["qa"] = "Question and Answer Session"
SECTION_TITLES["slides"] = "Presentation Slides"
SECTION_TITLES["front_matter"] = "Front Matter"

# The moderator's opening of the question session. Broad, because only the
# moderator says these and the moderator says little else.
_MODERATOR_QA_OPENERS = [
    "question and answer session",
    "question-and-answer session",
    "begin the question",
    "first question",
    "open the floor",
    "question queue",
    "for q&a",
]

# Management's hand-off to the moderator. Narrow, because these are matched
# against every speaker's turn, and an analyst prefacing "my first question"
# must not be read as the start of the session they are already in.
_HANDOVER_PHRASES = [
    "hand it over for q",
    "hand over for q",
    "hand it over to the operator",
    "hand it back to the operator",
    "hand it over for question",
    "open it up for q",
    "open it up for question",
    "open the floor for q",
    "open the line for q",
    "turn it over for q",
    "back to the operator",
    "over to the moderator",
]

_MODERATOR_NAMES = {"moderator", "operator", "host"}

# Labels that are followed by a colon on a transcript's cover page but are not
# speakers. Without this, "Sub: Transcript of the Analyst Meeting" opens a turn
# attributed to a person named "Sub", and the roster block becomes a turn by
# "MANAGEMENT" whose remarks are the list of participants.
_NON_SPEAKER_LABELS = {
    "sub", "subject", "re", "ref", "reference", "encl", "enclosure", "enclosures",
    "management", "participants", "present", "attendees", "date", "time", "venue",
    "agenda", "note", "notes", "cin", "tel", "telephone", "fax", "email", "e-mail",
    "website", "address", "regards", "thanking you", "from", "to", "cc", "copy",
    "isin", "scrip code", "symbol", "series", "disclaimer", "safe harbor",
    "safe harbour", "moderator instructions", "company secretary",
    "compliance officer", "yours faithfully", "yours sincerely",
}

# The sign-off block of a covering letter: "For <Issuer> Limited". The issuer's
# name cannot be enumerated, so the shape is matched instead -- "For" followed
# by a short phrase ending in a corporate suffix. Without this, the letter that
# fronts every exchange-filed transcript opens a turn attributed to a speaker
# named after the company.
_SIGNATURE_LABEL_RE = re.compile(
    r"^for\s+.{0,60}?\b(limited|ltd\.?|plc|inc\.?|corporation|corp\.?|company)$",
    re.IGNORECASE,
)

# "Srini Pallia:" or "MR. SRINI PALLIA:" at the start of a line. Up to four
# capitalised words keeps full Indian names while rejecting sentence fragments.
_SPEAKER_RE = re.compile(
    r"^(?:(?:Mr|Mrs|Ms|Dr)\.?\s+)?"
    r"((?:[A-Z][A-Za-z.'\-]*)(?:\s+[A-Z][A-Za-z.'\-]*){0,3})\s*:\s*(.*)$"
)

# The block of a transcript's cover page that names the participants.
_MANAGEMENT_HEADER_RE = re.compile(r"^\s*(MANAGEMENT|PARTICIPANTS|PRESENT)\s*:?\s*(.*)$",
                                   re.IGNORECASE)
_PERSON_ROLE_RE = re.compile(r"^(?:MR|MRS|MS|DR)\.?\s+([A-Z][A-Za-z.'\- ]+?)\s*[-,]\s*(.+)$",
                             re.IGNORECASE)

# The moderator introduces each questioner by name and firm.
_ANALYST_INTRO_RE = re.compile(
    r"from (?:the line of|)\s*([A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3})",
)

# One table cell: a figure, optionally bracketed, signed, or percentage, and
# optionally carrying a currency mark. Indian filings write the rupee as the
# sign, as "Rs." or as "INR" interchangeably, sometimes within one document. The
# integer part may not end in a comma, so that the "31," of a date is not a cell.
_FIGURE_TOKEN_RE = re.compile(
    r"^[(\[]?[-+]?(?:₹|Rs\.?|INR|US\$|\$|€|£)?\s?"
    r"\d(?:[\d,]*\d)?(?:\.\d+)?[)\]]?%?[*]?$",
    re.IGNORECASE,
)

# A currency mark standing alone as a token. Indian filings set the scale
# caption and the mark apart from the figure as often as they attach it.
_CURRENCY_TOKEN_RE = re.compile(r"^(?:₹|Rs\.?|INR|US\$|\$|€|£)$", re.IGNORECASE)

# Sentence-final punctuation, for deciding where a wrapped line ends a paragraph.
_SENTENCE_END = (".", "!", "?", ":", ";", '"', ")")

ROLE_MANAGEMENT: str = "management"
ROLE_ANALYST: str = "analyst"
ROLE_MODERATOR: str = "moderator"


# --- Data model ---------------------------------------------------------------

@dataclass
class Block:
    """One contiguous piece of a document.

    Attributes:
        kind: ``prose``, ``table``, ``heading``, or ``turn``.
        text: The block's text.
        page_start: First page the block draws from.
        page_end: Last page the block draws from.
        speaker: Speaker name, for transcript turns.
        role: Speaker role, for transcript turns.
        title: Slide or table title, where one applies.
    """

    kind: str
    text: str
    page_start: int
    page_end: int
    speaker: str = ""
    role: str = ""
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form, omitting empty optional fields."""
        out: Dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
        for name in ("speaker", "role", "title"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Block":
        """Rebuilds a Block from its JSON form."""
        return cls(
            kind=str(payload.get("kind", "prose")),
            text=str(payload.get("text", "")),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            speaker=str(payload.get("speaker", "")),
            role=str(payload.get("role", "")),
            title=str(payload.get("title", "")),
        )


@dataclass
class Section:
    """A run of a document that belongs together.

    Attributes:
        section_id: Canonical section key.
        title: Display title.
        page_start: First page of the section.
        page_end: Last page of the section.
        blocks: The section's blocks in reading order.
    """

    section_id: str
    title: str
    page_start: int
    page_end: int
    blocks: List[Block] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        """Total characters across the section's blocks."""
        return sum(len(block.text) for block in self.blocks)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form of the section."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "n_chars": self.n_chars,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Section":
        """Rebuilds a Section from its JSON form."""
        return cls(
            section_id=str(payload.get("section_id", "other")),
            title=str(payload.get("title", "")),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            blocks=[Block.from_dict(b) for b in (payload.get("blocks") or [])],
        )


@dataclass
class ParsedDocument:
    """A parsed filing.

    Attributes:
        doc_id: Registry identifier.
        doc_type: Document class.
        ticker: Owning ticker.
        label: Period label.
        n_pages: Pages read.
        sections: Sections in reading order.
        meta: Parser version and per-section counters.
    """

    doc_id: str
    doc_type: str
    ticker: str
    label: str
    n_pages: int = 0
    sections: List[Section] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        """Total characters across every section."""
        return sum(section.n_chars for section in self.sections)

    @property
    def n_blocks(self) -> int:
        """Total blocks across every section."""
        return sum(len(section.blocks) for section in self.sections)

    def section_sizes(self) -> Dict[str, int]:
        """Returns characters per section key, summed over repeated sections."""
        out: Dict[str, int] = {}
        for section in self.sections:
            out[section.section_id] = out.get(section.section_id, 0) + section.n_chars
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form written to the parse cache."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "ticker": self.ticker,
            "label": self.label,
            "parser_version": parser_version(self.doc_type),
            "n_pages": self.n_pages,
            "n_chars": self.n_chars,
            "n_blocks": self.n_blocks,
            "section_sizes": self.section_sizes(),
            "meta": self.meta,
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParsedDocument":
        """Rebuilds a ParsedDocument from its cached JSON form."""
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            ticker=str(payload.get("ticker", "")),
            label=str(payload.get("label", "")),
            n_pages=int(payload.get("n_pages", 0)),
            sections=[Section.from_dict(s) for s in (payload.get("sections") or [])],
            meta=dict(payload.get("meta") or {}),
        )


# --- Shared helpers -----------------------------------------------------------

def is_table_line(line: str) -> bool:
    """Returns True when a line reads as a row of a numeric table.

    Tables must not be reflowed into paragraphs: joining a statement of profit
    and loss into prose destroys the row-to-figure correspondence that makes it
    readable at all. Two or more figures on a short line is the signature.
    """
    tokens = line.split()
    if len(tokens) < 2:
        return False

    # Position, not density, is what separates the two cases. A table row ends
    # in its figures however long its label runs -- "Net Change in Cash and
    # Cash Equivalents (25,367) 24,733 (50,100)" is only a third figures -- and
    # prose that happens to carry figures does not: "For the quarter ended
    # December 31, 2025" ends in one, and the comma after the day stops the run.
    # A currency mark set off as its own token belongs to the figure beside it,
    # not to the label: "Total income INR 12,345 INR 11,200" is a table row, and
    # counting "INR" as a non-figure would break the trailing run after one.
    trailing = 0
    for token in reversed(tokens):
        if _CURRENCY_TOKEN_RE.match(token):
            continue
        if not _FIGURE_TOKEN_RE.match(token):
            break
        trailing += 1
    if trailing >= 2:
        return True

    # A row of bare figures with no label at all: a chart's data labels, or a
    # column header band split off from the rows beneath it.
    cells = [token for token in tokens if not _CURRENCY_TOKEN_RE.match(token)]
    figures = sum(1 for token in cells if _FIGURE_TOKEN_RE.match(token))
    return figures >= 3 and cells and figures >= 0.6 * len(cells)


def join_wrapped_lines(lines: Sequence[str]) -> List[str]:
    """Reflows hard-wrapped PDF lines into paragraphs.

    A PDF line break is a typesetting artifact, not a sentence boundary; left
    in place it truncates the sentence a chunk ends on. Table rows are passed
    through unchanged, because their line structure is their meaning.

    Args:
        lines: Lines of one page or block in reading order.

    Returns:
        Paragraphs and table rows, in order.
    """
    out: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        """Emits the buffered paragraph."""
        if buffer:
            out.append(" ".join(buffer).strip())
            del buffer[:]

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if is_table_line(line):
            flush()
            out.append(line)
            continue
        if not buffer:
            buffer.append(line)
            continue
        previous = buffer[-1]
        if previous.endswith("-") and len(previous) > 1 and previous[-2].isalpha():
            # A word hyphenated across the line break rejoins without a space.
            buffer[-1] = previous[:-1] + line
            continue
        starts_new = (
            previous.endswith(_SENTENCE_END)
            and (line[0].isupper() or line[0].isdigit() or not line[0].isalnum())
        )
        if starts_new:
            flush()
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return [item for item in out if item]


def blocks_from_lines(lines: Sequence[str], page: int) -> List[Block]:
    """Turns one page's lines into prose and table blocks.

    Consecutive table rows are gathered into a single table block so that a
    statement is not scattered across chunks one row at a time.
    """
    blocks: List[Block] = []
    table_rows: List[str] = []

    def flush_table() -> None:
        """Emits the buffered table rows as one block."""
        if table_rows:
            blocks.append(Block(kind="table", text="\n".join(table_rows),
                                page_start=page, page_end=page))
            del table_rows[:]

    for item in join_wrapped_lines(lines):
        if is_table_line(item):
            table_rows.append(item)
            continue
        flush_table()
        blocks.append(Block(kind="prose", text=item, page_start=page, page_end=page))
    flush_table()
    return blocks


# --- Annual reports -----------------------------------------------------------

def _heading_lines(page: PageText) -> List[str]:
    """Returns the lines of a page that read as headings.

    A heading is short, is not a sentence, and is set in caps or title case.
    Testing that shape is what lets a section title be recognised wherever the
    reading order happens to place it, without letting body prose announce a
    section change on the strength of a passing keyword.

    Args:
        page: An extracted page.

    Returns:
        The heading-like lines, in reading order.
    """
    out: List[str] = []
    for line in page.lines[:HEADING_SCAN_LINES]:
        text = line.strip().rstrip(":").strip()
        words = text.split()
        if not words or len(words) > MAX_HEADING_WORDS:
            continue
        if text.endswith((".", ";", ",")):
            continue
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 6:
            continue
        capitalised = sum(1 for word in words if word[:1].isupper())
        if text.isupper() or capitalised >= max(2, int(0.7 * len(words))):
            out.append(text)
    return out


def company_keywords(company: str) -> Dict[str, List[str]]:
    """Builds section keywords from the issuer's own name.

    Indian annual reports title their overview section after the company --
    "About <name>", "<name> at a glance" -- so the heading identifying the
    business section differs for every issuer. Deriving it from the catalogued
    company name keeps the classifier general: nothing here is tied to a
    particular filer, and a company whose name is unknown falls back on the
    generic headings alone.

    The legal suffix is stripped first. "About Limited" is not a heading, and
    matching a suffix alone would fire on any page that mentions one.

    Args:
        company: Company name as catalogued, or empty.

    Returns:
        Section key to the extra phrases to look for.
    """
    words = [
        word for word in (company or "").lower().replace(",", " ").split()
        if word.strip(".") not in _COMPANY_SUFFIXES
    ]
    if not words:
        return {}

    # Both the full short name and its leading word: filings head pages with
    # either the registered short name or just its first word.
    names = {" ".join(words)}
    if len(words) > 1 and len(words[0]) > 3:
        names.add(words[0])

    phrases = [
        template % name
        for name in sorted(names)
        for template in _COMPANY_HEADING_TEMPLATES
    ]
    return {"business": phrases}


def _score_section(
    text: str,
    weight: float,
    scores: Dict[str, float],
    extra: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Adds keyword hits in `text` to `scores`, scaled by `weight`."""
    lowered = text.lower()
    for key, _title, phrases in ANNUAL_SECTIONS:
        for phrase in phrases:
            if phrase in lowered:
                scores[key] = scores.get(key, 0.0) + weight
    for key, phrases in (extra or {}).items():
        for phrase in phrases:
            if phrase in lowered:
                scores[key] = scores.get(key, 0.0) + weight


def classify_page(
    page: PageText,
    extra_keywords: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, float]:
    """Classifies one annual-report page into a canonical section.

    The running head carries the most weight: issuers print the current
    section's name at the top of every page, which states the answer directly.
    The opening lines come next, since a section's first page leads with its
    heading, and the body text last, where a passing mention of "risk" must not
    outvote either.

    Args:
        page: An extracted page.
        extra_keywords: Issuer-specific phrases from `company_keywords`.

    Returns:
        The section key and its score. A zero score means the page announces no
        section and the caller should carry the previous one forward.
    """
    # A section change is *announced* -- by a running head or by a heading on
    # the page. Body text only ever confirms which of the announced candidates
    # it is, and can never switch sections on its own.
    #
    # The governance vocabulary has to include words that appear on every page
    # of a statutory block -- "committee", "remuneration", "annexure" -- and
    # five such mentions at a body weight of 0.4 would clear any switching
    # threshold on their own. Separating the two roles removes that hazard;
    # measured against the reports tested here it changed no assignment, so
    # treat it as a guard rather than as a fix for anything observed.
    announced: Dict[str, float] = {}
    _score_section(page.header, 6.0, announced, extra_keywords)
    # Headings are looked for anywhere on the page, not just in its first few
    # lines. The XY-cut emits the running head and any narrow sidebar column
    # before the body, so a section's own title routinely lands well down the
    # reading order: one issuer prints "MANAGEMENT DISCUSSION AND ANALYSIS
    # REPORT:" as the ninth line of its page. Restricting the scan to the top of
    # the page left that whole section labelled as the one before it.
    # Only the first heading or two may announce. A section begins at its title;
    # a heading further down the page is a sub-heading inside it, and letting
    # those announce fragments an integrated report badly -- a "Climate Action"
    # panel inside a business narrative moved whole runs of pages into the ESG
    # section, which the default skip list then dropped.
    _score_section("\n".join(_heading_lines(page)[:ANNOUNCING_HEADINGS]),
                   2.5, announced, extra_keywords)
    if not announced:
        return "", 0.0

    if len(announced) >= MAX_ANNOUNCED_SECTIONS:
        # A page announcing this many sections is a table of contents, not the
        # start of any of them. Picking the highest score there would relabel
        # the document at its index page.
        return "", 0.0

    confirming: Dict[str, float] = {}
    _score_section(page.text[:2500], 0.4, confirming, extra_keywords)
    best = max(
        announced.items(),
        key=lambda item: (item[1], confirming.get(item[0], 0.0)),
    )
    return best[0], best[1]


def assign_sections(
    pages: Sequence[PageText],
    min_run: int = 2,
    company: str = "",
) -> List[str]:
    """Assigns a section key to every page of an annual report.

    Pages with no signal inherit the section of the page before them, which is
    how a section's interior pages -- tables, photographs, continuation prose --
    end up correctly labelled. A run shorter than `min_run` is then absorbed
    into its neighbour: one passing mention of "risk management" inside the
    board report should not carve a one-page risk section out of it.

    Args:
        pages: Extracted pages in document order.
        min_run: Shortest run of pages allowed to stand as its own section.
        company: Company name, used to recognise the issuer's own headings.

    Returns:
        One section key per page.
    """
    extra = company_keywords(company)
    raw: List[str] = []
    current = ""
    for page in pages:
        key, score = classify_page(page, extra)
        if key and score >= 2.0:
            current = key
        elif not current:
            current = "front_matter"
        raw.append(current)

    # Absorb runs too short to be real sections.
    smoothed = list(raw)
    index = 0
    while index < len(smoothed):
        end = index
        while end + 1 < len(smoothed) and smoothed[end + 1] == smoothed[index]:
            end += 1
        length = end - index + 1
        if length < min_run and index > 0:
            replacement = smoothed[index - 1]
            for position in range(index, end + 1):
                smoothed[position] = replacement
        index = end + 1
    return smoothed


def parse_annual_report(
    pages: Sequence[PageText],
    doc_id: str,
    ticker: str,
    label: str,
    company: str = "",
) -> ParsedDocument:
    """Parses an annual report into classified sections.

    Args:
        pages: Extracted pages in document order.
        doc_id: Registry identifier.
        ticker: Owning ticker.
        label: Period label, e.g. ``FY2026``.
        company: Company name, used to recognise headings the issuer builds
            from its own name.

    Returns:
        The parsed document.
    """
    keys = assign_sections(pages, company=company)
    document = ParsedDocument(
        doc_id=doc_id, doc_type="annual_report", ticker=ticker, label=label,
        n_pages=len(pages),
    )

    current: Optional[Section] = None
    for page, key in zip(pages, keys):
        if current is None or current.section_id != key:
            if current is not None and current.blocks:
                document.sections.append(current)
            current = Section(
                section_id=key,
                title=SECTION_TITLES.get(key, key.replace("_", " ").title()),
                page_start=page.number,
                page_end=page.number,
            )
        current.page_end = page.number
        current.blocks.extend(blocks_from_lines(page.lines, page.number))
    if current is not None and current.blocks:
        document.sections.append(current)

    document.meta = {
        "sections_detected": sorted({section.section_id for section in document.sections}),
        "spread_pages": sum(1 for page in pages if page.is_spread),
        "multi_column_pages": sum(1 for page in pages if page.n_columns > 1),
        "table_blocks": sum(
            1 for section in document.sections
            for block in section.blocks if block.kind == "table"
        ),
    }
    return document


# --- Transcripts --------------------------------------------------------------

def _management_names(lines: Sequence[str]) -> Dict[str, str]:
    """Extracts the management roster printed on a transcript's cover page.

    Returns:
        Mapping of lower-cased speaker name to the role text as printed. An
        empty mapping is normal: not every issuer prints a roster, and the
        caller falls back to positional inference.
    """
    roster: Dict[str, str] = {}
    collecting = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        header = _MANAGEMENT_HEADER_RE.match(line)
        if header:
            collecting = True
            line = header.group(2).strip()
            if not line:
                continue
        if not collecting:
            continue
        match = _PERSON_ROLE_RE.match(line)
        if match:
            name = " ".join(match.group(1).split()).title()
            roster[name.lower()] = " ".join(match.group(2).split())
            continue
        if line.lower().startswith(("moderator", "operator")):
            break
        if len(roster) and not line.upper().startswith(("MR", "MRS", "MS", "DR")):
            # The roster block has ended.
            break
    return roster


def _find_qa_start(turns: Sequence[Tuple[str, str]]) -> Tuple[int, str]:
    """Locates the boundary between prepared remarks and the Q&A.

    The handover is spoken by whoever happens to hold the call. On some calls
    the CFO ends the prepared remarks with "with that, I will hand it over for
    Q&A" and the moderator's own opening never appears as a separate turn; on
    others the moderator announces the session in full. Both forms are matched,
    and each is matched only against the speakers who use it, so an analyst's
    "my first question" cannot open a session already under way.

    When neither form is found the structure of the call is used as a fallback:
    a moderator introduces the call, management delivers the remarks, and the
    moderator's next turn is the handover. That is a guess, so which method
    produced the boundary is recorded rather than assumed.

    Args:
        turns: Speaker turns in order, as (speaker, text) pairs.

    Returns:
        The index of the first Q&A turn, and the method that found it. An index
        equal to the number of turns means no Q&A was identified.
    """
    for index, (speaker, text) in enumerate(turns):
        lowered = text.lower()
        if speaker.lower() in _MODERATOR_NAMES:
            if any(opener in lowered for opener in _MODERATOR_QA_OPENERS):
                return index, "moderator_opening"
        elif any(phrase in lowered for phrase in _HANDOVER_PHRASES):
            # The hand-off itself belongs to the remarks; the Q&A starts next.
            return index + 1, "management_handover"

    moderator_turns = [
        index for index, (speaker, _text) in enumerate(turns)
        if speaker.lower() in _MODERATOR_NAMES
    ]
    if len(moderator_turns) >= 2:
        return moderator_turns[1], "second_moderator_turn"
    return len(turns), "none"


def _is_speaker_name(name: str) -> bool:
    """Returns True when a colon-terminated label names a person on the call.

    A transcript's cover page is full of ``Label: value`` lines, and a filing
    letter's "Sub:" would otherwise be attributed as a speaker whose remarks
    are the subject line. Requiring either a known moderator label or a
    multi-word personal name rejects the whole class.
    """
    cleaned = " ".join(name.split())
    if not cleaned or len(cleaned) > 40:
        return False
    lowered = cleaned.lower()
    if lowered in _MODERATOR_NAMES:
        return True
    if lowered in _NON_SPEAKER_LABELS:
        return False
    if _SIGNATURE_LABEL_RE.match(cleaned):
        return False
    return len(cleaned.split()) >= 2


def split_speaker_turns(lines: Sequence[str], pages: Sequence[int]) -> List[Tuple[str, str, int]]:
    """Splits transcript lines into speaker turns.

    Args:
        lines: Lines of the whole transcript in order.
        pages: Page number for each line, parallel to `lines`.

    Returns:
        Tuples of (speaker, text, first page). Text before the first attributed
        line is returned under an empty speaker, which is where the cover
        letter and the participant roster land.
    """
    turns: List[Tuple[str, str, int]] = []
    speaker = ""
    buffer: List[str] = []
    page = pages[0] if pages else 0

    def flush() -> None:
        """Emits the buffered turn."""
        if buffer:
            text = " ".join(join_wrapped_lines(buffer)).strip()
            if text:
                turns.append((speaker, text, page))
            del buffer[:]

    for line, line_page in zip(lines, pages):
        match = _SPEAKER_RE.match(line.strip())
        if match and _is_speaker_name(match.group(1)):
            flush()
            speaker = " ".join(match.group(1).split())
            page = line_page
            remainder = match.group(2).strip()
            if remainder:
                buffer.append(remainder)
            continue
        buffer.append(line)
    flush()
    return turns


def assign_roles(
    turns: Sequence[Tuple[str, str, int]],
    qa_start: int,
    roster: Dict[str, str],
) -> Dict[str, str]:
    """Infers each speaker's role.

    Three signals are combined, strongest first: the roster printed on the
    cover page, the moderator's introduction of each questioner by name, and
    position -- anyone speaking during prepared remarks is on the company's
    side of the call. The distinction matters downstream: a claim made by
    management is the company's assertion, while an analyst's question is
    evidence of what the market doubts, and a research report must not quote
    one as the other.

    Args:
        turns: Speaker turns in order.
        qa_start: Index of the first Q&A turn.
        roster: Management roster from the cover page.

    Returns:
        Mapping of speaker name to role.
    """
    roles: Dict[str, str] = {}
    for speaker, _text, _page in turns:
        if not speaker:
            continue
        lowered = speaker.lower()
        if lowered in _MODERATOR_NAMES:
            roles[speaker] = ROLE_MODERATOR
        elif lowered in roster:
            roles[speaker] = ROLE_MANAGEMENT

    for index, (speaker, text, _page) in enumerate(turns):
        if speaker and speaker.lower() in _MODERATOR_NAMES and index >= qa_start:
            for name in _ANALYST_INTRO_RE.findall(text):
                cleaned = " ".join(str(name).split())
                if cleaned and cleaned not in roles:
                    roles[cleaned] = ROLE_ANALYST

    for index, (speaker, _text, _page) in enumerate(turns):
        if not speaker or speaker in roles:
            continue
        roles[speaker] = ROLE_MANAGEMENT if index < qa_start else ROLE_ANALYST
    return roles


def parse_transcript(
    pages: Sequence[PageText],
    doc_id: str,
    ticker: str,
    label: str,
    company: str = "",
) -> ParsedDocument:
    """Parses a concall transcript into prepared remarks and Q&A.

    Args:
        pages: Extracted pages in document order.
        doc_id: Registry identifier.
        ticker: Owning ticker.
        label: Period label, e.g. ``Jul 2026``.
        company: Accepted for a uniform parser signature; a transcript's
            structure is the dialogue itself and does not depend on the name.

    Returns:
        The parsed document, with one block per speaker turn.
    """
    lines: List[str] = []
    line_pages: List[int] = []
    for page in pages:
        for line in page.lines:
            lines.append(line)
            line_pages.append(page.number)

    roster = _management_names(lines[:120])
    turns = split_speaker_turns(lines, line_pages)
    attributed = [(speaker, text) for speaker, text, _page in turns]
    qa_start, qa_method = _find_qa_start(attributed)
    roles = assign_roles(turns, qa_start, roster)

    document = ParsedDocument(
        doc_id=doc_id, doc_type="concall_transcript", ticker=ticker, label=label,
        n_pages=len(pages),
    )

    front = Section(section_id="front_matter", title=SECTION_TITLES["front_matter"],
                    page_start=pages[0].number if pages else 0,
                    page_end=pages[0].number if pages else 0)
    remarks = Section(section_id="prepared_remarks", title=SECTION_TITLES["prepared_remarks"],
                      page_start=0, page_end=0)
    qa = Section(section_id="qa", title=SECTION_TITLES["qa"], page_start=0, page_end=0)

    for index, (speaker, text, page) in enumerate(turns):
        block = Block(
            kind="turn" if speaker else "prose",
            text=text,
            page_start=page,
            page_end=page,
            speaker=speaker,
            role=roles.get(speaker, ""),
        )
        if not speaker and index == 0:
            target = front
        elif index < qa_start:
            target = remarks
        else:
            target = qa
        if not target.blocks:
            target.page_start = page
        target.page_end = max(target.page_end, page)
        target.blocks.append(block)

    for section in (front, remarks, qa):
        if section.blocks:
            document.sections.append(section)

    speakers = sorted({speaker for speaker, _t, _p in turns if speaker})
    document.meta = {
        "turns": len([t for t in turns if t[0]]),
        "qa_turns": max(0, len(turns) - qa_start),
        "speakers": speakers,
        "roles": {name: roles.get(name, "") for name in speakers},
        "roster_found": bool(roster),
        "qa_boundary_found": qa_start < len(turns),
        "qa_boundary_method": qa_method,
    }
    if qa_method == "none" and turns:
        logger.info("[%s] no question-and-answer handover found; "
                    "all turns filed as prepared remarks.", doc_id)
    elif qa_method == "second_moderator_turn":
        logger.info("[%s] no handover phrase found; Q&A boundary inferred from "
                    "the moderator's second turn.", doc_id)
    return document


# --- Presentations ------------------------------------------------------------

def parse_presentation(
    pages: Sequence[PageText],
    doc_id: str,
    ticker: str,
    label: str,
    company: str = "",
) -> ParsedDocument:
    """Parses an investor presentation into one block group per slide.

    A slide's title is often the only thing that says what its figures measure,
    so it is attached to every block cut from that slide.

    Args:
        pages: Extracted pages in document order.
        doc_id: Registry identifier.
        ticker: Owning ticker.
        label: Period label.
        company: Accepted for a uniform parser signature; slide titles carry
            their own context and do not depend on the name.

    Returns:
        The parsed document.
    """
    document = ParsedDocument(
        doc_id=doc_id, doc_type="concall_presentation", ticker=ticker, label=label,
        n_pages=len(pages),
    )
    section = Section(
        section_id="slides", title=SECTION_TITLES["slides"],
        page_start=pages[0].number if pages else 0,
        page_end=pages[-1].number if pages else 0,
    )
    titles: List[str] = []
    for page in pages:
        if not page.lines:
            continue
        title = normalize_text(page.lines[0])[:120].strip()
        titles.append(title)
        for block in blocks_from_lines(page.lines, page.number):
            block.title = title
            section.blocks.append(block)
    if section.blocks:
        document.sections.append(section)
    document.meta = {
        "slides": len(titles),
        "slide_titles": titles,
        "table_blocks": sum(1 for block in section.blocks if block.kind == "table"),
    }
    return document


# --- Entry point --------------------------------------------------------------

PARSERS = {
    "annual_report": parse_annual_report,
    "concall_transcript": parse_transcript,
    "concall_presentation": parse_presentation,
}


def parse_document(
    path: Path,
    doc_id: str,
    doc_type: str,
    ticker: str,
    label: str,
    max_pages: Optional[int] = None,
    company: str = "",
) -> ParsedDocument:
    """Extracts and parses one filing according to its class.

    Args:
        path: PDF on disk.
        doc_id: Registry identifier.
        doc_type: One of the keys of PARSERS.
        ticker: Owning ticker.
        label: Period label.
        max_pages: Optional cap on pages read, for smoke tests.
        company: Company name as catalogued, used by the annual-report section
            classifier to recognise headings built from the issuer's own name.

    Returns:
        The parsed document. A PDF that yields no text produces a document with
        no sections rather than raising, so the registry can record the failure
        against that one filing.

    Raises:
        KeyError: If `doc_type` has no parser.
    """
    parser = PARSERS[doc_type]
    pages = extract_pdf_pages(path, max_pages=max_pages)
    if not pages:
        logger.error("[%s] no pages extracted from %s", doc_id, path)
        return ParsedDocument(doc_id=doc_id, doc_type=doc_type, ticker=ticker, label=label)
    document = parser(pages, doc_id, ticker, label, company)
    logger.info("[%s] parsed %d pages into %d sections, %d blocks, %d chars",
                doc_id, document.n_pages, len(document.sections),
                document.n_blocks, document.n_chars)
    return document
