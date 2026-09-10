"""The data model for everything extracted from a filing.

This is the boundary between "read the PDF" and "decide what a chunk is": the
extractor fills it, the chunker consumes it, and it is cached to disk in between
so that re-chunking a 500-page annual report never means re-reading the PDF.

Three decisions here shape what retrieval can later do:

* **A block carries its heading trail** (`Block.path`, outermost first), so a
  chunker can say "this paragraph is inside Management Discussion, under Segment
  Review" from the document's own structure instead of guessing from keywords.
* **A table stays a table.** Header rows are kept apart from the body, so the
  header can be re-attached to every fragment the table is split into. A row
  reading ``Revenue from operations 926,163 790,935`` without its header is not
  evidence -- nothing in it says which column is which year, and that is the
  commonest way a financial table becomes unusable downstream.
* **A figure is a file plus what is known about it**: the image beside the
  document, its caption, and the classifier verdict, so a later vision pass can
  tell a bar chart from a logo.

Every dataclass round-trips through `to_dict`/`from_dict`: this is a cache
format as much as an in-memory model.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Block kinds. Kept small on purpose: a chunker that branches on twenty kinds is
# a chunker with twenty untested paths.
KIND_HEADING: str = "heading"
KIND_TEXT: str = "text"
KIND_LIST: str = "list"
KIND_TABLE: str = "table"
KIND_FIGURE: str = "figure"
KIND_CAPTION: str = "caption"
KIND_FORMULA: str = "formula"
KIND_CODE: str = "code"

KINDS: tuple[str, ...] = (
    KIND_HEADING,
    KIND_TEXT,
    KIND_LIST,
    KIND_TABLE,
    KIND_FIGURE,
    KIND_CAPTION,
    KIND_FORMULA,
    KIND_CODE,
)


def _clean_cell(value: Any) -> str:
    """Returns a value as a single line: a newline in a cell breaks markdown."""
    return " ".join(("" if value is None else str(value)).split())


def _present(**fields: Any) -> dict[str, Any]:
    """Returns the fields that carry a value, for an optional-field to_dict."""
    return {name: value for name, value in fields.items() if value}


@dataclass
class Table:
    """One table, with its structure intact.

    Attributes:
        rows: Every row, header rows included.
        header_rows: How many leading rows are header. Zero means the source had
            no recognisable header -- worth recording rather than papering over.
        caption: Caption as printed, when one was found.
        page: Page the table starts on.

    """

    rows: list[list[str]] = field(default_factory=list)
    header_rows: int = 0
    caption: str = ""
    page: int = 0

    @property
    def n_rows(self) -> int:
        """Total rows, header included."""
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        """Width of the widest row."""
        return max((len(row) for row in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        """Returns the header as one flat row, or [] when absent.

        Multi-row headers are joined column-wise: a two-deep header ("Revenue |
        FY2026" over "| Rs m") only means anything as one label per column.
        """
        if self.header_rows <= 0:
            return []
        return [
            " ".join(
                row[column]
                for row in self.rows[: self.header_rows]
                if column < len(row) and row[column]
            )
            for column in range(self.n_cols)
        ]

    @property
    def body(self) -> list[list[str]]:
        """Returns the rows below the header."""
        return self.rows[self.header_rows :]

    def to_markdown(self, include_caption: bool = True) -> str:
        """Renders the table as GitHub-flavoured markdown.

        Pipes rather than a text dump: the delimiters keep the row-to-column
        correspondence legible to a model reading the chunk.

        Args:
            include_caption: Prefix the caption when there is one. It carries
                the scale ("Rs in millions"), so a table rendered without it can
                be read off by a factor of a million.

        Returns:
            The markdown table, or "" when there are no rows.

        """
        if not self.rows:
            return ""
        width = self.n_cols

        def render(cells: list[str]) -> str:
            padded = list(cells) + [""] * (width - len(cells))
            return "| " + " | ".join(cell.replace("|", "\\|") for cell in padded) + " |"

        separator = "|" + "|".join([" --- "] * width) + "|"
        header = self.header
        # With no header to anchor the columns every row is body, but the
        # separator is still emitted so the result stays valid markdown.
        lines = [render(header), separator] if header else [separator]
        lines.extend(render(row) for row in (self.body if header else self.rows))

        table = "\n".join(lines)
        if include_caption and self.caption:
            return self.caption + "\n" + table
        return table

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON form, omitting empty optional fields."""
        return {
            "rows": self.rows,
            "header_rows": self.header_rows,
            **_present(caption=self.caption, page=self.page),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Table:
        """Rebuilds a Table from its JSON form."""
        return cls(
            rows=[
                [_clean_cell(cell) for cell in (row or [])]
                for row in (payload.get("rows") or [])
            ],
            header_rows=int(payload.get("header_rows", 0)),
            caption=str(payload.get("caption", "")),
            page=int(payload.get("page", 0)),
        )


@dataclass
class Figure:
    """One image or chart lifted out of a filing.

    Attributes:
        path: Image location relative to the stock directory, e.g.
            ``figures/annual_report_FY2026/p0142-03.png``. Relative so that
            moving the output directory does not invalidate it.
        caption: Caption as printed, when one was found.
        kind: What the picture classifier called it (``bar_chart``, ``logo``),
            or "" when classification did not run. Decides which figures are
            worth a vision pass and which are page furniture.
        page: Page the figure was cropped from.
        width: Pixel width of the saved image.
        height: Pixel height of the saved image.

    """

    path: str = ""
    caption: str = ""
    kind: str = ""
    page: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON form, omitting empty optional fields."""
        out: dict[str, Any] = {
            "path": self.path,
            "page": self.page,
            **_present(caption=self.caption, kind=self.kind),
        }
        if self.width or self.height:
            # Written as a pair: one zero dimension still means something when
            # the other is known.
            out["width"], out["height"] = self.width, self.height
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Figure:
        """Rebuilds a Figure from its JSON form."""
        return cls(
            path=str(payload.get("path", "")),
            caption=str(payload.get("caption", "")),
            kind=str(payload.get("kind", "")),
            page=int(payload.get("page", 0)),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
        )


@dataclass
class Block:
    """One element of a filing, in reading order.

    Attributes:
        kind: One of KINDS.
        text: The block's text. For a table this is the markdown rendering, so a
            consumer needing only text never has to know about tables; `table`
            carries the structure for one that does.
        page: Page the block starts on.
        level: Heading depth, 1 for the outermost; 0 for anything not a heading.
        path: The headings this block sits under, outermost first -- the field
            phase-2 sectioning is built on.
        table: Structure, when `kind` is a table.
        figure: Image reference, when `kind` is a figure.

    """

    kind: str
    text: str = ""
    page: int = 0
    level: int = 0
    path: list[str] = field(default_factory=list)
    table: Table | None = None
    figure: Figure | None = None

    @property
    def n_chars(self) -> int:
        """Length of the block's text."""
        return len(self.text)

    @property
    def section(self) -> str:
        """Returns the outermost heading this block sits under, or ''."""
        return self.path[0] if self.path else ""

    @property
    def heading_trail(self) -> str:
        """Returns the heading path as one readable line."""
        return " > ".join(self.path)

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON form, omitting empty optional fields."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "page": self.page,
            **_present(level=self.level, path=list(self.path)),
        }
        if self.table is not None:
            out["table"] = self.table.to_dict()
        if self.figure is not None:
            out["figure"] = self.figure.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Block:
        """Rebuilds a Block from its JSON form."""
        table, figure = payload.get("table"), payload.get("figure")
        return cls(
            kind=str(payload.get("kind", KIND_TEXT)),
            text=str(payload.get("text", "")),
            page=int(payload.get("page", 0)),
            level=int(payload.get("level", 0)),
            path=[str(item) for item in (payload.get("path") or [])],
            table=Table.from_dict(table) if isinstance(table, dict) else None,
            figure=Figure.from_dict(figure) if isinstance(figure, dict) else None,
        )


@dataclass
class ExtractedDocument:
    """Everything recovered from one filing.

    Attributes:
        doc_id: Registry identifier.
        doc_type: Document class, as catalogued.
        ticker: Owning ticker.
        label: Period label, e.g. ``FY2026``.
        n_pages: Pages the extractor produced content for.
        n_source_pages: Pages the PDF itself declares. Held beside `n_pages` so
            that a silent partial failure -- a text PDF with forty image-only
            pages inside it -- is visible rather than merely absent.
        extractor: Version tag of the code that produced this.
        blocks: Every block, in reading order.
        meta: Counters and extractor settings, for diagnosis.

    """

    doc_id: str
    doc_type: str = ""
    ticker: str = ""
    label: str = ""
    n_pages: int = 0
    n_source_pages: int = 0
    extractor: str = ""
    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- Derived views -------------------------------------------------------

    @property
    def n_chars(self) -> int:
        """Total characters across every block."""
        return sum(block.n_chars for block in self.blocks)

    @property
    def tables(self) -> list[Table]:
        """Every table, in document order."""
        return [b.table for b in self.blocks if b.table is not None]

    @property
    def figures(self) -> list[Figure]:
        """Every figure, in document order."""
        return [b.figure for b in self.blocks if b.figure is not None]

    @property
    def headings(self) -> list[Block]:
        """Every heading block, in document order."""
        return [b for b in self.blocks if b.kind == KIND_HEADING]

    def kind_counts(self) -> dict[str, int]:
        """Returns the number of blocks per kind."""
        return dict(Counter(block.kind for block in self.blocks))

    def blocks_by_page(self) -> dict[int, int]:
        """Returns the number of blocks recovered per page."""
        return dict(Counter(block.page for block in self.blocks))

    def chars_by_page(self) -> dict[int, int]:
        """Returns characters of text recovered per page."""
        tally: Counter[int] = Counter()
        for block in self.blocks:
            tally[block.page] += block.n_chars
        return dict(tally)

    def empty_pages(self) -> list[int]:
        """Returns the declared pages that yielded no content of any kind.

        Blocks, not characters, are what count: a slide holding one chart and no
        prose was extracted successfully, so counting characters would report
        every page of an investor deck as a failure. This looks for the page that
        produced *nothing* -- a scanned insert no text extractor recovers, which
        is otherwise indistinguishable from a page that was genuinely blank.
        """
        seen = self.blocks_by_page()
        total = self.n_source_pages or self.n_pages
        return [page for page in range(1, total + 1) if seen.get(page, 0) == 0]

    def text_free_pages(self) -> list[int]:
        """Returns pages that produced blocks but no text.

        Ordinary for a deck of charts, a red flag for an annual report -- which
        is why it is reported separately from `empty_pages` rather than folded
        into it.
        """
        chars = self.chars_by_page()
        return sorted(
            page
            for page, count in self.blocks_by_page().items()
            if count and chars.get(page, 0) == 0
        )

    def sections(self) -> list[str]:
        """Returns the distinct outermost headings, in first-seen order."""
        return list(
            dict.fromkeys(block.section for block in self.blocks if block.section)
        )

    # --- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON form written to the extraction cache.

        The counters are denormalised alongside the blocks so that a cache file
        can be judged -- pages, blocks, tables, sections -- without parsing every
        block back into an object.
        """
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "ticker": self.ticker,
            "label": self.label,
            "extractor": self.extractor,
            "n_pages": self.n_pages,
            "n_source_pages": self.n_source_pages,
            "n_chars": self.n_chars,
            "n_blocks": len(self.blocks),
            "kind_counts": self.kind_counts(),
            "n_tables": len(self.tables),
            "n_figures": len(self.figures),
            "sections": self.sections(),
            "meta": self.meta,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtractedDocument:
        """Rebuilds an ExtractedDocument from its JSON form."""
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            ticker=str(payload.get("ticker", "")),
            label=str(payload.get("label", "")),
            n_pages=int(payload.get("n_pages", 0)),
            n_source_pages=int(payload.get("n_source_pages", 0)),
            extractor=str(payload.get("extractor", "")),
            blocks=[Block.from_dict(item) for item in (payload.get("blocks") or [])],
            meta=dict(payload.get("meta") or {}),
        )

    def to_markdown(self) -> str:
        """Renders the whole document as markdown, for eyeballing an extraction.

        Not used by the pipeline. It exists because the fastest way to judge an
        extraction is to read it, and reading JSON blocks is not reading.
        """
        parts: list[str] = []
        for block in self.blocks:
            if block.kind == KIND_HEADING:
                parts.append("#" * max(1, min(6, block.level or 1)) + " " + block.text)
            elif block.kind == KIND_FIGURE and block.figure is not None:
                label = block.figure.caption or block.figure.kind or "figure"
                parts.append(f"![{label}]({block.figure.path})")
            else:
                parts.append(block.text)
        return "\n\n".join(part for part in parts if part.strip())
