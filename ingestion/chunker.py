"""Section-aware chunking of parsed filings, with a disk cache.

Chunking decides what a retrieval hit can possibly say, so the boundaries are
drawn along the document's own structure rather than at a fixed character
count. Blocks are packed into chunks up to a target size, but a chunk never
crosses a section boundary, never mixes two speakers' turns, and never splits a
table away from the rows it belongs to. The result is that every chunk can be
cited: it has a section, a page range, and -- for a transcript -- a named
speaker whose role is known.

Each chunk also carries a **context header** into its embedding text. An
isolated paragraph reading "margin declined 120 basis points sequentially" is
nearly unretrievable, because nothing in it says which company, which period,
or which document said so; prefixing the ticker, document class, period, and
section is what makes the vector answer the question actually being asked.

The cache is keyed by a fingerprint over the chunk texts, which is what lets
the embedding stage tell whether its input has really changed. Re-running with
the same parameters produces byte-identical chunks and therefore no
re-embedding.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ingestion.parsers import Block, ParsedDocument, Section
from ingestion.registry import sha256_text

logger = logging.getLogger(__name__)

# Bumped whenever a change here would produce different chunks from the same
# parse output. The registry compares this against what it recorded, so a
# behaviour change that leaves the version alone is invisible: the chunks on
# disk stay in use and the vectors keep describing text the chunker would no
# longer produce.
#   v2: overlap no longer reaches table blocks or speaker turns.
#   v3: a short trailing block is merged only into a preceding prose chunk.
#   v4: chunks with no substantive content are dropped rather than stored.
CHUNKER_VERSION: str = "chunker/v4"

# Target chunk size in characters. Roughly 250-350 tokens, which keeps a whole
# argument in one chunk while staying well inside the embedding model's window.
TARGET_CHARS: int = 1400

# A chunk is never emitted below this unless it is all that its section holds.
MIN_CHARS: int = 220

# Hard ceiling. A single block longer than this is split at sentence bounds.
MAX_CHARS: int = 2600

# Characters of the preceding chunk repeated at the start of the next one, so a
# sentence spanning a boundary is retrievable from either side.
OVERLAP_CHARS: int = 180

# Floors for a chunk to be worth storing. These are hygiene, not semantics:
# they exist to keep fragments with no content at all out of the store, and are
# set at the point where they stop dropping real sentences. "Revenue for the
# year grew to 926,163 million on the back of large deal wins" is fourteen
# words and fifty-eight letters, so the floors sit below that; a courtesy turn
# of similar length will still get through, and it is the query side -- which no
# longer mirrors the chunk's context header -- that stops such a turn ranking
# above real evidence. A table row needs only a label, because "Operating
# Activities 149,316 169,426 (20,110)" is nineteen letters and is exactly the
# evidence a report wants.
MIN_PROSE_LETTERS: int = 45
MIN_PROSE_WORDS: int = 12
MIN_TABLE_LETTERS: int = 8

# Document classes whose blocks must never be packed together, because the
# identity of the speaker is part of the evidence.
_SPEAKER_ISOLATED_KINDS = {"turn"}

# Annual-report sections left out of the chunk set by default.
#
# These are not low-value in general -- they are low-value *here*. Three
# quarters of a modern integrated annual report is compliance material, and
# embedding it costs most of the ingest's wall clock while making retrieval
# worse rather than better:
#
# * `financials` is the audited statements and the notes to accounts. Every
#   figure in them is already published by the data provider and typeset by the
#   deterministic reporting engine, which can verify its own arithmetic. A
#   language model retrieving a figure from note 34 cannot, and 1,954 of the
#   2,579 chunks of one large-cap integrated report measured here were exactly
#   that -- competing for top-k against the ~230 chunks of narrative that a
#   research report is actually written from.
# * `notice` is the AGM notice and e-voting instructions.
# * `governance` is the statutory governance and remuneration report.
# * `esg` is the BRSR and sustainability report -- material for an ESG mandate,
#   and retrievable by passing an empty --skip-sections, but not part of a
#   fundamental equity thesis.
#
# Pass `--skip-sections` on the CLI to override, including with nothing at all
# to embed the whole volume.
DEFAULT_ANNUAL_SKIP_SECTIONS: List[str] = ["financials", "notice", "governance", "esg"]


@dataclass
class Chunk:
    """One retrievable unit of a filing.

    Attributes:
        chunk_id: Stable identifier: ``<doc_id>::<section_id>::<index>``.
        ticker: Owning ticker.
        doc_id: Source document.
        doc_type: Source document class.
        label: Period label of the source document.
        section_id: Canonical section key.
        section_title: Display title of the section.
        text: The chunk's text as it appears in the filing.
        kind: ``prose``, ``table``, or ``turn``.
        page_start: First page the chunk draws from.
        page_end: Last page the chunk draws from.
        speaker: Speaker name, for transcript turns.
        role: Speaker role, for transcript turns.
        title: Slide title, for presentation chunks.
        n_chars: Length of `text`.
    """

    chunk_id: str
    ticker: str
    doc_id: str
    doc_type: str
    label: str
    section_id: str
    section_title: str
    text: str
    kind: str = "prose"
    page_start: int = 0
    page_end: int = 0
    speaker: str = ""
    role: str = ""
    title: str = ""

    @property
    def n_chars(self) -> int:
        """Length of the chunk's text."""
        return len(self.text)

    @property
    def context_header(self) -> str:
        """Returns the provenance line prepended to the embedding text.

        This is the difference between a vector that means "some company's
        margin fell" and one that means "this issuer's services margin fell in
        the June 2026 quarter, said by the CFO on the earnings call".
        """
        parts = [self.ticker, DOC_TYPE_LABELS.get(self.doc_type, self.doc_type), self.label]
        if self.section_title:
            parts.append(self.section_title)
        if self.title:
            parts.append(self.title)
        if self.speaker:
            parts.append("%s (%s)" % (self.speaker, self.role or "speaker"))
        return " | ".join(part for part in parts if part)

    @property
    def embed_text(self) -> str:
        """Returns the exact text submitted to the embedding model."""
        return self.context_header + "\n" + self.text

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form written to the chunk cache."""
        out: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "ticker": self.ticker,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "label": self.label,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "kind": self.kind,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "n_chars": self.n_chars,
            "text": self.text,
        }
        for name in ("speaker", "role", "title"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        """Rebuilds a Chunk from its cached JSON form."""
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            ticker=str(payload.get("ticker", "")),
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            label=str(payload.get("label", "")),
            section_id=str(payload.get("section_id", "")),
            section_title=str(payload.get("section_title", "")),
            text=str(payload.get("text", "")),
            kind=str(payload.get("kind", "prose")),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            speaker=str(payload.get("speaker", "")),
            role=str(payload.get("role", "")),
            title=str(payload.get("title", "")),
        )


DOC_TYPE_LABELS: Dict[str, str] = {
    "annual_report": "Annual Report",
    "concall_transcript": "Earnings Call Transcript",
    "concall_presentation": "Investor Presentation",
}


@dataclass
class ChunkSet:
    """Every chunk cut from one document, plus its identity.

    Attributes:
        doc_id: Source document.
        doc_type: Source document class.
        ticker: Owning ticker.
        label: Period label.
        params: Chunker parameters used, recorded so a parameter change
            invalidates the cache.
        chunks: The chunks in document order.
    """

    doc_id: str
    doc_type: str
    ticker: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Chunk] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """Returns a hash over every chunk's identity and text.

        This is the embedding stage's staleness key. It changes when the
        document's content changes, when the parser changes what it extracts,
        or when the chunker's parameters change -- and only then.
        """
        payload = "\n".join(chunk.chunk_id + "\x1f" + chunk.embed_text for chunk in self.chunks)
        return sha256_text(CHUNKER_VERSION + "\x1e" + payload)

    def section_counts(self) -> Dict[str, int]:
        """Returns the number of chunks per section."""
        out: Dict[str, int] = {}
        for chunk in self.chunks:
            out[chunk.section_id] = out.get(chunk.section_id, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form written to the chunk cache."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "ticker": self.ticker,
            "label": self.label,
            "chunker_version": CHUNKER_VERSION,
            "params": self.params,
            "fingerprint": self.fingerprint,
            "count": len(self.chunks),
            "section_counts": self.section_counts(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkSet":
        """Rebuilds a ChunkSet from its cached JSON form."""
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            ticker=str(payload.get("ticker", "")),
            label=str(payload.get("label", "")),
            params=dict(payload.get("params") or {}),
            chunks=[Chunk.from_dict(item) for item in (payload.get("chunks") or [])],
        )


def has_substance(text: str, kind: str) -> bool:
    """Returns True when a chunk carries enough content to be worth retrieving.

    A chunk's embedding text is its context header followed by its content, so a
    chunk with almost no content is almost pure header -- and scores like one.
    An annual report yields chunks whose entire text is ``2 2`` -- chart data
    labels stranded from their axis -- and an earnings call yields turns that
    read ``Thank you so much and all the very best``. Left in the store they do not
    merely waste space: they are the *closest* match to any query, because the
    header is all they have and the header is what every query shares. Dropping
    them at chunk time is the only place the fix belongs, since a retrieval-time
    filter would still have to rank them first to exclude them.

    Args:
        text: The chunk's text.
        kind: Block kind the chunk was built from.

    Returns:
        True when the chunk should be kept.
    """
    letters = sum(1 for character in text if character.isalpha())
    if kind == "table":
        # A table row needs a label. Rows of bare figures are chart data labels
        # or a column band severed from its rows, and neither can be cited.
        return letters >= MIN_TABLE_LETTERS
    return letters >= MIN_PROSE_LETTERS and len(text.split()) >= MIN_PROSE_WORDS


def _split_long_text(text: str, limit: int) -> List[str]:
    """Splits an oversized block at sentence boundaries.

    A single paragraph longer than the ceiling is rare in prose but ordinary in
    a transcript, where one answer can run to several thousand characters.
    Splitting mid-sentence would leave both halves quoting a fragment, so the
    split is made at the last sentence end before the limit, and only falls
    back to a hard cut when a single sentence exceeds it.

    Args:
        text: Block text.
        limit: Maximum characters per piece.

    Returns:
        Pieces of at most `limit` characters, in order.
    """
    if len(text) <= limit:
        return [text]

    pieces: List[str] = []
    remainder = text
    while len(remainder) > limit:
        window = remainder[:limit]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if cut < limit // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        else:
            cut += 1
        pieces.append(remainder[:cut].strip())
        remainder = remainder[cut:].strip()
    if remainder:
        pieces.append(remainder)
    return [piece for piece in pieces if piece]


def _tail(text: str, size: int) -> str:
    """Returns the last whole sentence or phrase of `text`, up to `size`."""
    if size <= 0 or not text:
        return ""
    window = text[-size:]
    for marker in (". ", "? ", "! "):
        position = window.find(marker)
        if 0 <= position < len(window) - 2:
            return window[position + 2:].strip()
    space = window.find(" ")
    return window[space + 1:].strip() if space >= 0 else window.strip()


def chunk_section(
    section: Section,
    document: ParsedDocument,
    index_start: int,
    target_chars: int = TARGET_CHARS,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[Chunk]:
    """Cuts one section into chunks.

    Args:
        section: Section to chunk.
        document: Parent document, for provenance fields.
        index_start: First chunk index within the document.
        target_chars: Size a chunk is packed towards.
        min_chars: Smallest chunk emitted while more blocks remain.
        max_chars: Hard ceiling per chunk.
        overlap_chars: Characters of the previous chunk repeated at the start.

    Returns:
        The section's chunks in order.
    """
    chunks: List[Chunk] = []
    buffer: List[Block] = []
    index = index_start
    carry = ""

    def emit() -> None:
        """Emits the buffered blocks as one chunk."""
        nonlocal index, carry
        if not buffer:
            return
        body = "\n\n".join(block.text for block in buffer).strip()
        if not body:
            del buffer[:]
            return
        lead = buffer[0]
        # Overlap exists so that a sentence spanning a boundary is retrievable
        # from either side, which only applies to running prose. Prepending the
        # previous chunk's tail to a table would caption a statement of profit
        # and loss with an unrelated paragraph, and prepending it to a speaker's
        # turn would attribute the previous speaker's words to this one.
        standalone = lead.kind in _SPEAKER_ISOLATED_KINDS or lead.kind == "table"
        text = (carry + "\n\n" + body).strip() if carry and not standalone else body
        chunks.append(Chunk(
            chunk_id="%s::%s::%03d" % (document.doc_id, section.section_id, index),
            ticker=document.ticker,
            doc_id=document.doc_id,
            doc_type=document.doc_type,
            label=document.label,
            section_id=section.section_id,
            section_title=section.title,
            text=text,
            kind=lead.kind if len(set(b.kind for b in buffer)) == 1 else "prose",
            page_start=min(block.page_start for block in buffer),
            page_end=max(block.page_end for block in buffer),
            speaker=lead.speaker,
            role=lead.role,
            title=lead.title,
        ))
        index += 1
        # A standalone block also carries nothing forward: the next prose chunk
        # should overlap the prose before it, not a table that interrupted it.
        carry = "" if standalone else _tail(body, overlap_chars)
        del buffer[:]

    def buffered_chars() -> int:
        """Characters currently buffered, including block separators."""
        return sum(len(block.text) + 2 for block in buffer)

    for block in section.blocks:
        # A speaker's turn is its own evidence and is never packed with the
        # next speaker's; the same holds for a table, whose rows must not be
        # prefixed with unrelated prose.
        isolated = block.kind in _SPEAKER_ISOLATED_KINDS or block.kind == "table"
        if isolated and buffer:
            emit()

        pieces = _split_long_text(block.text, max_chars)
        for position, piece in enumerate(pieces):
            part = Block(
                kind=block.kind,
                text=piece,
                page_start=block.page_start,
                page_end=block.page_end,
                speaker=block.speaker,
                role=block.role,
                title=block.title,
            )
            buffer.append(part)
            if isolated or buffered_chars() >= target_chars or position < len(pieces) - 1:
                emit()

    if not buffer:
        return chunks

    mergeable = (
        chunks
        and chunks[-1].kind == "prose"
        and buffer[0].kind == "prose"
        and buffered_chars() < min_chars
    )
    if mergeable:
        # A short prose tail joins the prose chunk before it rather than
        # standing alone, where it would be too small to retrieve on its own
        # terms. It is never merged into a table or a speaker's turn: that would
        # append unrelated commentary to a statement, or one speaker's words to
        # another's, which is the same defect the overlap rule avoids.
        tail = "\n\n".join(block.text for block in buffer).strip()
        if tail:
            chunks[-1].text = (chunks[-1].text + "\n\n" + tail).strip()
            chunks[-1].page_end = max(chunks[-1].page_end,
                                      max(block.page_end for block in buffer))
        del buffer[:]
    else:
        emit()
    return chunks


def chunk_document(
    document: ParsedDocument,
    target_chars: int = TARGET_CHARS,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    skip_sections: Optional[Iterable[str]] = None,
) -> ChunkSet:
    """Cuts a parsed document into chunks.

    Args:
        document: Parsed document.
        target_chars: Size a chunk is packed towards.
        min_chars: Smallest chunk emitted while more blocks remain.
        max_chars: Hard ceiling per chunk.
        overlap_chars: Characters repeated across a chunk boundary.
        skip_sections: Section keys to leave out entirely.

    Returns:
        The document's ChunkSet.
    """
    skip = set(skip_sections or ())
    chunk_set = ChunkSet(
        doc_id=document.doc_id,
        doc_type=document.doc_type,
        ticker=document.ticker,
        label=document.label,
        params={
            "target_chars": target_chars,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "skip_sections": sorted(skip),
            "min_prose_letters": MIN_PROSE_LETTERS,
            "min_prose_words": MIN_PROSE_WORDS,
            "min_table_letters": MIN_TABLE_LETTERS,
        },
    )
    index = 0
    dropped = 0
    for section in document.sections:
        if section.section_id in skip:
            continue
        section_chunks = chunk_section(
            section, document, index,
            target_chars=target_chars, min_chars=min_chars,
            max_chars=max_chars, overlap_chars=overlap_chars,
        )
        kept = [c for c in section_chunks if has_substance(c.text, c.kind)]
        dropped += len(section_chunks) - len(kept)
        chunk_set.chunks.extend(kept)
        index += len(section_chunks)

    logger.info("[%s] chunked into %d chunks (%s)%s",
                document.doc_id, len(chunk_set.chunks),
                ", ".join("%s=%d" % item for item in sorted(chunk_set.section_counts().items())),
                "; %d dropped as insubstantial" % dropped if dropped else "")
    return chunk_set


def write_chunk_cache(chunk_set: ChunkSet, path: Path) -> None:
    """Writes a ChunkSet to its cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunk_set.to_dict(), ensure_ascii=False), encoding="utf-8")


def read_chunk_cache(path: Path) -> Optional[ChunkSet]:
    """Reads a ChunkSet from its cache file, or None when unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Chunk cache unreadable at %s: %s", path, exc)
        return None
    return ChunkSet.from_dict(payload)


def write_parse_cache(document: ParsedDocument, path: Path) -> None:
    """Writes a parsed document to its cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document.to_dict(), ensure_ascii=False), encoding="utf-8")


def read_parse_cache(path: Path) -> Optional[ParsedDocument]:
    """Reads a parsed document from its cache file, or None when unusable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Parse cache unreadable at %s: %s", path, exc)
        return None
    return ParsedDocument.from_dict(payload)
