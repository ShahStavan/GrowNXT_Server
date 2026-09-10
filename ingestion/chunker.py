"""Multimodal PDF Element Chunker using LlamaIndex.

Follows the CocoIndex PDF Elements architecture (https://cocoindex.io/blogs/pdf-elements/)
by transforming extracted document blocks into native LlamaIndex nodes:
1. Text: Processed via LlamaIndex `SentenceSplitter` into `TextNode`s.
2. Tables: Preserved as structured Markdown `TextNode`s with repeated header rows.
3. Figures: Transformed into `ImageNode`s with image paths, captions, and dimensions.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    Document as LlamaDocument,
    ImageNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)

from ingestion.documents.content import (
    KIND_FIGURE,
    KIND_HEADING,
    KIND_TABLE,
    ExtractedDocument,
    Figure,
    Table,
)

logger = logging.getLogger(__name__)

CHUNKER_VERSION: str = "chunk/llama-elements-v1"
ELEMENT_TEXT: str = "text"
ELEMENT_TABLE: str = "table"
ELEMENT_FIGURE: str = "figure"
ELEMENT_TYPES: tuple[str, ...] = (ELEMENT_TEXT, ELEMENT_TABLE, ELEMENT_FIGURE)

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100
MAX_TABLE_ROWS: int = 30
MAX_TABLE_CHARS: int = 2500
MIN_FIGURE_PIXELS: int = 48


def _sha256(text: str) -> str:
    """Returns the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """Multimodal chunk unit fully interoperable with LlamaIndex BaseNode."""

    chunk_id: str
    doc_id: str
    ticker: str
    doc_type: str = ""
    label: str = ""
    element_type: str = ELEMENT_TEXT
    page_start: int = 0
    page_end: int = 0
    section: str = ""
    path: list[str] = field(default_factory=list)
    text: str = ""
    embed_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    figure_path: str = ""
    table: Table | None = None
    figure: Figure | None = None

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def heading_trail(self) -> str:
        return " > ".join(self.path) if self.path else ""

    def to_node(self) -> BaseNode:
        """Converts to a native LlamaIndex TextNode or ImageNode."""
        meta = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "label": self.label,
            "element_type": self.element_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "heading_path": list(self.path),
            "heading_trail": self.heading_trail,
            **self.metadata,
        }
        if self.element_type == ELEMENT_FIGURE and self.figure_path:
            return ImageNode(
                id_=self.chunk_id,
                image_path=self.figure_path,
                text=self.text,
                metadata=meta,
                excluded_embed_metadata_keys=["figure_path", "width", "height"],
            )
        return TextNode(
            id_=self.chunk_id,
            text=self.text,
            metadata=meta,
            excluded_embed_metadata_keys=["chunk_id", "doc_id"],
        )

    @classmethod
    def from_node(cls, node: BaseNode) -> Chunk:
        """Builds a Chunk from a LlamaIndex Node."""
        meta = dict(node.metadata or {})
        fig_path = str(meta.get("figure_path", ""))
        if isinstance(node, ImageNode) and node.image_path:
            fig_path = node.image_path

        return cls(
            chunk_id=str(node.node_id),
            doc_id=str(meta.get("doc_id", "")),
            ticker=str(meta.get("ticker", "")),
            doc_type=str(meta.get("doc_type", "")),
            label=str(meta.get("label", "")),
            element_type=str(meta.get("element_type", ELEMENT_TEXT)),
            page_start=int(meta.get("page_start", 0)),
            page_end=int(meta.get("page_end", 0)),
            section=str(meta.get("section", "")),
            path=[str(p) for p in (meta.get("heading_path") or [])],
            text=node.get_content(metadata_mode=MetadataMode.NONE),
            embed_text=node.get_content(metadata_mode=MetadataMode.EMBED),
            metadata=meta,
            figure_path=fig_path,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "label": self.label,
            "element_type": self.element_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "path": list(self.path),
            "text": self.text,
            "embed_text": self.embed_text,
            "metadata": self.metadata,
        }
        if self.figure_path:
            out["figure_path"] = self.figure_path
        if self.table is not None:
            out["table"] = self.table.to_dict()
        if self.figure is not None:
            out["figure"] = self.figure.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        tbl = payload.get("table")
        fig = payload.get("figure")
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            ticker=str(payload.get("ticker", "")),
            doc_type=str(payload.get("doc_type", "")),
            label=str(payload.get("label", "")),
            element_type=str(payload.get("element_type", ELEMENT_TEXT)),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            section=str(payload.get("section", "")),
            path=[str(p) for p in (payload.get("path") or [])],
            text=str(payload.get("text", "")),
            embed_text=str(payload.get("embed_text", "")),
            metadata=dict(payload.get("metadata") or {}),
            figure_path=str(payload.get("figure_path", "")),
            table=Table.from_dict(tbl) if isinstance(tbl, dict) else None,
            figure=Figure.from_dict(fig) if isinstance(fig, dict) else None,
        )


@dataclass
class ChunkSet:
    """Collection of chunks produced from a corporate filing."""

    doc_id: str
    ticker: str
    doc_type: str = ""
    label: str = ""
    fingerprint: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def n_chars(self) -> int:
        return sum(c.n_chars for c in self.chunks)

    def element_counts(self) -> dict[str, int]:
        return dict(Counter(c.element_type for c in self.chunks))

    def chunks_by_page(self) -> dict[int, int]:
        return dict(Counter(c.page_start for c in self.chunks if c.page_start))

    def section_counts(self) -> dict[str, int]:
        return dict(Counter(c.section for c in self.chunks if c.section))

    def to_nodes(self) -> list[BaseNode]:
        """Converts to native LlamaIndex nodes linked with PREVIOUS / NEXT relationships."""
        nodes = [c.to_node() for c in self.chunks]
        for i in range(len(nodes)):
            if i > 0:
                nodes[i].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=nodes[i - 1].node_id
                )
            if i < len(nodes) - 1:
                nodes[i].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=nodes[i + 1].node_id
                )
        return nodes

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[BaseNode],
        doc_id: str = "",
        ticker: str = "",
        doc_type: str = "",
        label: str = "",
        params: dict[str, Any] | None = None,
    ) -> ChunkSet:
        """Reconstructs a ChunkSet from a sequence of LlamaIndex nodes."""
        chunks = [Chunk.from_node(n) for n in nodes]
        fingerprint = _sha256(
            "|".join(f"{c.chunk_id}:{_sha256(c.text)}" for c in chunks)
            + f"|{json.dumps(params or {}, sort_keys=True)}"
        )
        return cls(
            doc_id=doc_id or (chunks[0].doc_id if chunks else ""),
            ticker=ticker or (chunks[0].ticker if chunks else ""),
            doc_type=doc_type or (chunks[0].doc_type if chunks else ""),
            label=label or (chunks[0].label if chunks else ""),
            fingerprint=fingerprint,
            params=params or {},
            chunks=chunks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "label": self.label,
            "fingerprint": self.fingerprint,
            "n_chunks": self.n_chunks,
            "n_chars": self.n_chars,
            "element_counts": self.element_counts(),
            "section_counts": self.section_counts(),
            "params": self.params,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChunkSet:
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            ticker=str(payload.get("ticker", "")),
            doc_type=str(payload.get("doc_type", "")),
            label=str(payload.get("label", "")),
            fingerprint=str(payload.get("fingerprint", "")),
            params=dict(payload.get("params") or {}),
            chunks=[Chunk.from_dict(item) for item in (payload.get("chunks") or [])],
        )

    def to_markdown(self) -> str:
        parts: list[str] = [
            f"# Chunks for {self.ticker} - {self.doc_id} ({self.doc_type})",
            f"**Total Chunks:** {self.n_chunks} | **Elements:** {self.element_counts()}\n",
        ]
        for idx, chunk in enumerate(self.chunks, 1):
            tag = f"[{chunk.element_type.upper()}]"
            trail = chunk.heading_trail or "(top-level)"
            parts.append(
                f"### Chunk {idx}: {tag} Page {chunk.page_start} | {trail}\n"
                f"*ID:* `{chunk.chunk_id}`\n\n{chunk.text}\n\n---"
            )
        return "\n\n".join(parts)


def chunk_params(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    skip_sections: Sequence[str] | None = None,
    min_figure_pixels: int = MIN_FIGURE_PIXELS,
) -> dict[str, Any]:
    """Returns the settings that determine a chunk set's content.

    `chunk_document` records this on the `ChunkSet` and folds it into the
    fingerprint, so a caller wanting to know whether a chunk file on disk is
    what this configuration would produce compares this dict against the
    recorded one -- one comparison, not a list of fields that drifts the next
    time a setting is added here.

    Args:
        document: The extraction being chunked; its `extractor` version is
            part of the answer, because different text chunks differently.
        chunk_size: Target chunk size, in characters.
        chunk_overlap: Overlap between adjacent chunks, in characters.
        skip_sections: Section headings to drop.
        min_figure_pixels: Smallest figure edge worth keeping.

    Returns:
        The parameter record, JSON-round-trippable.
    """
    return {
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "skip_sections": sorted({s.strip().lower() for s in (skip_sections or ())}),
        "min_figure_pixels": int(min_figure_pixels),
        "version": CHUNKER_VERSION,
        # Recorded so the indexer's state can tell a chunk file produced by an
        # older extractor from one the current pipeline would produce.
        "extractor": str(document.extractor or ""),
    }


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    skip_sections: Sequence[str] | None = None,
    min_figure_pixels: int = MIN_FIGURE_PIXELS,
) -> ChunkSet:
    """Decomposes an ExtractedDocument into multimodal element chunks using LlamaIndex."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"[^,.;。？！]+[,.;。？！]?",
    )
    skip_set = {s.strip().lower() for s in (skip_sections or ())}
    chunks: list[Chunk] = []
    seq = 0

    buf_text: list[str] = []
    buf_pages: list[int] = []
    cur_path: list[str] = []
    cur_section: str = ""

    def _context(page: int, trail: str, tag: str = "") -> str:
        header_tag = f" | {tag}" if tag else ""
        trail_line = f"Section: {trail}\n\n" if trail else ""
        return (
            f"[{document.ticker} | {document.doc_type} | {document.label} | Page {page}{header_tag}]\n"
            f"{trail_line}"
        )

    def _flush_text() -> None:
        nonlocal buf_text, buf_pages, cur_path, cur_section, seq
        if not buf_text:
            return

        combined = "\n\n".join(buf_text).strip()
        if not combined:
            buf_text, buf_pages = [], []
            return

        p_start = min(buf_pages) if buf_pages else 0
        p_end = max(buf_pages) if buf_pages else p_start
        trail = " > ".join(cur_path) if cur_path else ""

        # LlamaIndex SentenceSplitter invocation
        doc_wrapper = LlamaDocument(
            text=combined,
            metadata={
                "doc_id": document.doc_id,
                "ticker": document.ticker,
                "doc_type": document.doc_type,
                "label": document.label,
                "page_start": p_start,
                "page_end": p_end,
                "section": cur_section,
                "heading_path": list(cur_path),
                "heading_trail": trail,
            },
        )
        for node in splitter.get_nodes_from_documents([doc_wrapper]):
            clean = node.get_content(metadata_mode=MetadataMode.NONE).strip()
            if len(clean) < 10:
                continue

            seq += 1
            cid = f"{document.doc_id}_p{p_start:04d}_txt_{seq:03d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=document.doc_id,
                    ticker=document.ticker,
                    doc_type=document.doc_type,
                    label=document.label,
                    element_type=ELEMENT_TEXT,
                    page_start=p_start,
                    page_end=p_end,
                    section=cur_section,
                    path=list(cur_path),
                    text=clean,
                    embed_text=_context(p_start, trail) + clean,
                    metadata={"char_count": len(clean), "heading_trail": trail},
                )
            )

        buf_text, buf_pages = [], []

    for block in document.blocks:
        sec = block.section or (block.path[0] if block.path else "")
        if sec and any(skip in sec.lower() for skip in skip_set):
            continue

        if block.kind == KIND_HEADING:
            _flush_text()
            cur_path = list(block.path)
            cur_section = sec
            continue

        if block.kind == KIND_TABLE:
            _flush_text()
            tbl = block.table
            page = block.page or (tbl.page if tbl else 0)
            trail = " > ".join(block.path) if block.path else ""

            if tbl is None or not tbl.rows:
                if block.text.strip():
                    seq += 1
                    cid = f"{document.doc_id}_p{page:04d}_tbl_{seq:03d}"
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=document.doc_id,
                            ticker=document.ticker,
                            doc_type=document.doc_type,
                            label=document.label,
                            element_type=ELEMENT_TABLE,
                            page_start=page,
                            page_end=page,
                            section=sec,
                            path=list(block.path),
                            text=block.text.strip(),
                            embed_text=_context(page, trail, "Table")
                            + block.text.strip(),
                            table=tbl,
                        )
                    )
                continue

            tbl_md = tbl.to_markdown(include_caption=True)
            if tbl.n_rows <= MAX_TABLE_ROWS and len(tbl_md) <= MAX_TABLE_CHARS:
                seq += 1
                cid = f"{document.doc_id}_p{page:04d}_tbl_{seq:03d}"
                meta = {
                    "n_rows": tbl.n_rows,
                    "n_cols": tbl.n_cols,
                    "header_rows": tbl.header_rows,
                    "caption": tbl.caption,
                }
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=document.doc_id,
                        ticker=document.ticker,
                        doc_type=document.doc_type,
                        label=document.label,
                        element_type=ELEMENT_TABLE,
                        page_start=page,
                        page_end=page,
                        section=sec,
                        path=list(block.path),
                        text=tbl_md,
                        embed_text=_context(page, trail, "Table") + tbl_md,
                        metadata=meta,
                        table=tbl,
                    )
                )
            else:
                # Row-wise partitioning with header preservation
                hdrs = tbl.rows[: tbl.header_rows] if tbl.header_rows > 0 else []
                body = tbl.body if tbl.header_rows > 0 else tbl.rows
                batch_size = max(10, MAX_TABLE_ROWS - max(1, tbl.header_rows))
                total_parts = (len(body) + batch_size - 1) // batch_size

                for start in range(0, len(body), batch_size):
                    sub_table = Table(
                        rows=hdrs + body[start : start + batch_size],
                        header_rows=len(hdrs),
                        caption=tbl.caption,
                        page=page,
                    )
                    sub_md = sub_table.to_markdown(include_caption=True)
                    part = (start // batch_size) + 1
                    seq += 1
                    cid = f"{document.doc_id}_p{page:04d}_tbl_{seq:03d}"
                    meta = {
                        "n_rows": sub_table.n_rows,
                        "n_cols": sub_table.n_cols,
                        "header_rows": sub_table.header_rows,
                        "caption": tbl.caption,
                        "part": part,
                        "total_parts": total_parts,
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=document.doc_id,
                            ticker=document.ticker,
                            doc_type=document.doc_type,
                            label=document.label,
                            element_type=ELEMENT_TABLE,
                            page_start=page,
                            page_end=page,
                            section=sec,
                            path=list(block.path),
                            text=sub_md,
                            embed_text=_context(
                                page, trail, f"Table Part {part}/{total_parts}"
                            )
                            + sub_md,
                            metadata=meta,
                            table=sub_table,
                        )
                    )
            continue

        if block.kind == KIND_FIGURE:
            _flush_text()
            fig = block.figure
            if (
                fig is None
                or fig.width < min_figure_pixels
                or fig.height < min_figure_pixels
            ):
                continue
            if fig.kind == "icon" and not fig.caption:
                continue

            page = fig.page or block.page
            trail = " > ".join(block.path) if block.path else ""
            kind_lbl = fig.kind.replace("_", " ").title() if fig.kind else "Figure"
            caption = fig.caption or f"{kind_lbl} on page {page}"

            seq += 1
            cid = f"{document.doc_id}_p{page:04d}_fig_{seq:03d}"
            text = (
                f"![{caption}]({fig.path})\n\n"
                f"**Figure Kind:** {fig.kind or 'chart'}\n"
                f"**Caption:** {fig.caption or 'None'}\n"
                f"**Resolution:** {fig.width}x{fig.height} px"
            )
            embed_text = f"{_context(page, trail, f'Figure: {fig.kind}')}Figure: {kind_lbl}. Caption: {fig.caption or 'Chart'}"

            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=document.doc_id,
                    ticker=document.ticker,
                    doc_type=document.doc_type,
                    label=document.label,
                    element_type=ELEMENT_FIGURE,
                    page_start=page,
                    page_end=page,
                    section=sec,
                    path=list(block.path),
                    text=text,
                    embed_text=embed_text,
                    figure_path=fig.path,
                    metadata={
                        "figure_path": fig.path,
                        "kind": fig.kind,
                        "caption": fig.caption,
                        "width": fig.width,
                        "height": fig.height,
                    },
                    figure=fig,
                )
            )
            continue

        # Standard prose text / list / formula / code
        if block.path != cur_path:
            _flush_text()
            cur_path = list(block.path)
            cur_section = sec

        content = block.text.strip()
        if content:
            buf_text.append(content)
            if block.page:
                buf_pages.append(block.page)
            if sum(len(t) for t in buf_text) >= chunk_size * 3:
                _flush_text()

    _flush_text()

    params = chunk_params(
        document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_sections=skip_sections,
        min_figure_pixels=min_figure_pixels,
    )
    fingerprint = _sha256(
        "|".join(f"{c.chunk_id}:{_sha256(c.text)}" for c in chunks)
        + f"|{json.dumps(params, sort_keys=True)}"
    )

    chunk_set = ChunkSet(
        doc_id=document.doc_id,
        ticker=document.ticker,
        doc_type=document.doc_type,
        label=document.label,
        fingerprint=fingerprint,
        params=params,
        chunks=chunks,
    )

    elem_counts = chunk_set.element_counts()
    logger.info(
        "[%s] chunked %s with LlamaIndex into %d chunks (%d text, %d tables, %d figures) in %d chars",
        document.ticker,
        document.doc_id,
        chunk_set.n_chunks,
        elem_counts.get(ELEMENT_TEXT, 0),
        elem_counts.get(ELEMENT_TABLE, 0),
        elem_counts.get(ELEMENT_FIGURE, 0),
        chunk_set.n_chars,
    )
    return chunk_set


def write_chunk_cache(chunk_set: ChunkSet, path: Path) -> None:
    """Writes a ChunkSet atomically to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(
        json.dumps(chunk_set.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(target)


def read_chunk_cache(path: Path) -> ChunkSet | None:
    """Reads a ChunkSet from disk, returning None if unreadable or absent."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return ChunkSet.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read chunk cache at %s: %s", path, exc)
        return None
