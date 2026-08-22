"""Docling-based extraction of a filing into text, tables and figures.

This replaces a hand-tuned pdfplumber stack -- word boxes, a recursive XY-cut
for multi-column pages, point thresholds for gutters and cells, a keyword
classifier over running heads -- all of which reconstructed from geometry what a
layout model reports directly, on thresholds calibrated against a handful of
reports.

What Docling supplies that the old stack could not:

* **Reading order from a layout model**, so a three-column page or a two-page
  spread comes out in the order a person reads it.
* **Table structure from TableFormer**, a grid with its header cells marked --
  the difference between evidence and a trap, since ``Revenue from operations
  926,163 790,935`` means nothing without the header naming the years.
* **A heading hierarchy**, so every block carries the trail it sits under: the
  document's own statement of its structure, and what the chunker sections on.
* **Figures as images, and OCR for pages that are pictures of text.** An
  investor deck is mostly charts and a scanned filing is entirely one; both used
  to yield nothing at all, silently.

Two operational notes. Model weights download once, on the first conversion,
into Docling's cache (``~/.cache/docling`` unless ``DOCLING_ARTIFACTS_PATH``
says otherwise). And this is the expensive stage -- a 500-page annual report is
minutes of CPU -- which is why its output is cached under ``extracted/``.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from ingestion.documents.content import (
    KIND_CAPTION,
    KIND_CODE,
    KIND_FIGURE,
    KIND_FORMULA,
    KIND_HEADING,
    KIND_LIST,
    KIND_TABLE,
    KIND_TEXT,
    Block,
    ExtractedDocument,
    Figure,
    Table,
)
from ingestion.documents.storage import DocumentStore

logger = logging.getLogger(__name__)

# Bumped whenever a change here would produce a different extraction from the
# same PDF. The registry compares this against what it recorded, so a behaviour
# change that leaves the version alone is invisible: the cache on disk stays in
# use and describes text this module would no longer produce.
EXTRACT_VERSION: str = "extract/docling-v1"

# Docling logs one line per page per model at INFO.
_QUIET_LOGGERS = ("docling", "docling_core", "docling_ibm_models", "PIL")

# Resolution multiplier for saved figures. 2.0 is roughly 144 dpi: enough for a
# chart's axis labels to stay legible to a later vision pass, without writing
# 30 MB of PNGs for one deck.
DEFAULT_IMAGE_SCALE: float = 2.0

# Pictures smaller than this in either dimension are page furniture -- rules,
# bullets, logo fragments -- and not worth a file on disk.
MIN_FIGURE_PIXELS: int = 48

# A guard rather than a policy: a "PDF" of 5,000 pages is a misdirected
# download, and Docling would spend an hour proving it.
MAX_PAGES: int = 2000

# Docling labels mapped onto this package's block kinds. Anything unmapped
# becomes prose, the safe default: a block wrongly called prose is still
# retrievable, one dropped is not.
_LABEL_KINDS: dict[str, str] = {
    "title": KIND_HEADING,
    "section_header": KIND_HEADING,
    "list_item": KIND_LIST,
    "caption": KIND_CAPTION,
    "formula": KIND_FORMULA,
    "code": KIND_CODE,
    "table": KIND_TABLE,
    "picture": KIND_FIGURE,
    "text": KIND_TEXT,
    "paragraph": KIND_TEXT,
    "footnote": KIND_TEXT,
    "page_header": KIND_TEXT,
    "page_footer": KIND_TEXT,
}

# Page furniture, kept out of the block stream: a running head repeated on 500
# pages adds 500 near-identical blocks that say nothing, and the heading
# hierarchy already carries what it was there to say.
_FURNITURE_LABELS = frozenset({"page_header", "page_footer"})


class ExtractionError(RuntimeError):
    """Raised when a filing cannot be converted at all."""


@dataclass
class Extractor:
    """Converts filings to `ExtractedDocument` with Docling.

    The converter is built on first use and reused, because constructing it
    loads the layout and table models -- seconds of work that must not be
    repeated per document.

    Attributes:
        ocr: Run OCR over pages with no extractable text. Costs real time, and
            is the only way a scanned filing yields anything.
        figures: Extract and save images.
        accurate_tables: Use TableFormer's accurate mode rather than its fast
            one. Slower per table, and the difference shows up precisely on the
            dense financial tables this pipeline cares about.
        cell_matching: Match recognised cells back to the PDF's own text, so the
            exact printed figures survive rather than being re-recognised.
        image_scale: Resolution multiplier for saved figures.
        max_pages: Cap on pages converted, for smoke tests.
    """

    ocr: bool = True
    figures: bool = True
    accurate_tables: bool = True
    cell_matching: bool = True
    image_scale: float = DEFAULT_IMAGE_SCALE
    max_pages: int | None = None
    _converter: Any = field(default=None, init=False, repr=False)

    # --- Conversion ----------------------------------------------------------

    @property
    def converter(self) -> Any:
        """Returns the Docling converter, building it on first use.

        Raises:
            ExtractionError: If Docling is not installed.
        """
        if self._converter is None:
            self._converter = self._build()
        return self._converter

    def _build(self) -> Any:
        """Constructs a `DocumentConverter` configured for filings."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - declared in requirements
            raise ExtractionError(
                "docling is not installed; run pip install -r requirements.txt"
            ) from exc

        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        options = PdfPipelineOptions()
        options.do_ocr = bool(self.ocr)
        options.do_table_structure = True
        options.table_structure_options.do_cell_matching = bool(self.cell_matching)
        options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if self.accurate_tables else TableFormerMode.FAST
        )
        for name, value in (
            # Full page images are not kept -- only the cropped figures -- because
            # asking for both doubles the memory a 500-page conversion holds.
            ("generate_page_images", False),
            ("generate_table_images", False),
            ("generate_picture_images", bool(self.figures)),
            ("images_scale", float(self.image_scale)),
            # Classification is what lets a later stage tell a bar chart from a
            # logo and spend a vision pass only on the former.
            ("do_picture_classification", bool(self.figures)),
        ):
            _set_if_present(options, name, value)

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def run(
        self,
        pdf: Path,
        store: DocumentStore,
        doc_id: str,
        doc_type: str = "",
        ticker: str = "",
        label: str = "",
        max_pages: int | None = None,
        write: bool = True,
    ) -> ExtractedDocument:
        """Extracts one filing and, by default, caches the result.

        Args:
            pdf: The PDF on disk.
            store: Store the figures and the extraction JSON are written into.
            doc_id: Registry identifier; names the cache and the figure folder.
            doc_type: Document class, carried through for downstream stages.
            ticker: Owning ticker.
            label: Period label.
            max_pages: Cap on pages for this call, overriding the instance's.
            write: Write the extraction JSON. False is for callers that only
                want the object, such as the verification harness.

        Returns:
            The extracted document. A filing that yields no blocks is returned
            empty rather than raised on, so the caller can record the failure
            against that one document and carry on with the rest.

        Raises:
            ExtractionError: If Docling is unavailable or the file cannot be
                opened as a PDF at all.
        """
        source = Path(pdf)
        if not source.exists():
            raise ExtractionError("no such file: %s" % source)

        limit = max_pages if max_pages is not None else self.max_pages
        started = time.monotonic()
        try:
            result = self.converter.convert(
                str(source),
                max_num_pages=int(limit) if limit else MAX_PAGES,
            )
        except ExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad filing is a failed document
            raise ExtractionError("docling could not convert %s: %s" % (source, exc)) from exc

        docling_doc = getattr(result, "document", None)
        if docling_doc is None:
            raise ExtractionError("docling returned no document for %s" % source)

        document = ExtractedDocument(
            doc_id=doc_id, doc_type=doc_type, ticker=ticker, label=label,
            extractor=EXTRACT_VERSION,
        )
        document.blocks = list(self._blocks(docling_doc, store, doc_id))
        document.n_source_pages = _source_pages(docling_doc, source)
        document.n_pages = len({block.page for block in document.blocks if block.page})
        document.meta = {
            "ocr": bool(self.ocr),
            "figures": bool(self.figures),
            "accurate_tables": bool(self.accurate_tables),
            "cell_matching": bool(self.cell_matching),
            "image_scale": float(self.image_scale),
            "max_pages": int(limit) if limit else 0,
            "seconds": round(time.monotonic() - started, 1),
        }
        self._report(document)

        if write:
            write_extraction(document, store.extraction(doc_id))
        return document

    # --- Block construction --------------------------------------------------

    def _blocks(
        self,
        docling_doc: Any,
        store: DocumentStore,
        doc_id: str,
    ) -> Iterable[Block]:
        """Walks a DoclingDocument in reading order, yielding blocks.

        A heading stack is maintained as the walk proceeds, so every block
        carries the trail above it. That trail is why this walk exists rather
        than a call to ``export_to_markdown``, which flattens the hierarchy into
        ``#`` characters and leaves recovering it a guessing game.
        """
        trail: list[tuple[int, str]] = []
        figures_on_page: Counter[int] = Counter()

        for item, depth in _iterate(docling_doc):
            label = _label_of(item)
            if label in _FURNITURE_LABELS:
                continue

            page = _page_of(item)
            kind = _LABEL_KINDS.get(label, KIND_TEXT)
            path = [name for _level, name in trail]

            if kind == KIND_TABLE:
                table = _table_of(item, docling_doc, page)
                if table is not None and table.rows:
                    yield Block(kind=KIND_TABLE, text=table.to_markdown(), page=page,
                                path=path, table=table)
                continue

            if kind == KIND_FIGURE:
                if not self.figures:
                    continue
                figures_on_page[page] += 1
                figure = self._save_figure(
                    item, docling_doc, store, doc_id, page, figures_on_page[page],
                )
                if figure is not None:
                    yield Block(kind=KIND_FIGURE, text=figure.caption, page=page,
                                path=path, figure=figure)
                continue

            text = _text_of(item)
            if not text:
                continue

            if kind == KIND_HEADING:
                level = _heading_level(item, label, depth)
                # Truncate to the parent depth, then push: a heading at the same
                # level replaces its sibling rather than nesting under it.
                trail = [entry for entry in trail if entry[0] < level]
                trail.append((level, text))
                yield Block(kind=KIND_HEADING, text=text, page=page, level=level,
                            path=[name for _level, name in trail])
                continue

            yield Block(kind=kind, text=text, page=page, path=path)

    def _save_figure(
        self,
        item: Any,
        docling_doc: Any,
        store: DocumentStore,
        doc_id: str,
        page: int,
        index: int,
    ) -> Figure | None:
        """Writes one picture to the store and returns its reference.

        Returns None for a picture that could not be rendered or is too small to
        be anything but furniture. A failure here is logged and skipped rather
        than raised: losing one logo must not cost the whole filing.
        """
        try:
            image = item.get_image(docling_doc)
        except Exception as exc:  # noqa: BLE001 - a picture is not worth the document
            logger.debug("[%s] could not render a picture on page %d: %s", doc_id, page, exc)
            image = None
        if image is None:
            return None

        width, height = int(getattr(image, "width", 0)), int(getattr(image, "height", 0))
        if width < MIN_FIGURE_PIXELS or height < MIN_FIGURE_PIXELS:
            return None

        target = store.figure(doc_id, page, index)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(target), format="PNG")
        except Exception as exc:  # noqa: BLE001 - as above
            logger.debug("[%s] could not save a picture on page %d: %s", doc_id, page, exc)
            return None

        return Figure(
            path=store.relative(target),
            caption=_caption_of(item, docling_doc),
            kind=_classification_of(item),
            page=page,
            width=width,
            height=height,
        )

    def _report(self, document: ExtractedDocument) -> None:
        """Logs what the extraction recovered, and what it did not.

        The second half matters more: an extraction that returns something for
        most pages looks successful, so the pages that produced nothing are
        named. A run of them inside an otherwise dense document is the signature
        of image-only pages -- a reason to enable OCR, not a property of the
        filing.
        """
        counts = document.kind_counts()
        logger.info(
            "[%s] extracted %d pages, %d blocks, %d chars (%s) in %ss",
            document.doc_id, document.n_pages, len(document.blocks), document.n_chars,
            ", ".join("%s=%d" % pair for pair in sorted(counts.items())) or "nothing",
            document.meta.get("seconds", "?"),
        )
        empty = document.empty_pages()
        if empty:
            logger.warning(
                "[%s] %d of %d pages yielded nothing at all (e.g. %s). A scanned "
                "insert needs OCR; without it those pages are silently absent "
                "from the index rather than reported missing.",
                document.doc_id, len(empty),
                document.n_source_pages or document.n_pages,
                ", ".join(str(page) for page in empty[:8]),
            )
        text_free = document.text_free_pages()
        if text_free:
            # Ordinary for a deck, notable for an annual report: the figures are
            # on disk but nothing reads them yet.
            logger.info(
                "[%s] %d page(s) produced figures or tables but no text (e.g. %s).",
                document.doc_id, len(text_free),
                ", ".join(str(page) for page in text_free[:8]),
            )


# --- Docling adapters ---------------------------------------------------------
#
# Narrow readers over Docling's object model, kept apart so an upgrade that moves
# an attribute is a one-line change here rather than a hunt through the walk
# above. Each degrades to a harmless default instead of raising: a filing that
# loses one caption is still a good extraction, one that raises is no extraction.

def _iterate(docling_doc: Any) -> Iterable[tuple[Any, int]]:
    """Yields (item, depth) over a DoclingDocument in reading order."""
    try:
        for entry in docling_doc.iterate_items():
            if isinstance(entry, tuple) and len(entry) == 2:
                yield entry[0], int(entry[1] or 0)
            else:
                yield entry, 0
    except Exception as exc:  # noqa: BLE001 - an unwalkable document is empty
        logger.error("Could not walk the converted document: %s", exc)


def _label_of(item: Any) -> str:
    """Returns an item's Docling label as a plain lower-case string."""
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label) or "").strip().lower()


def _text_of(item: Any) -> str:
    """Returns an item's text, whitespace-normalised."""
    return " ".join(str(getattr(item, "text", "") or "").split())


def _page_of(item: Any) -> int:
    """Returns the 1-based page an item sits on, or 0 when unknown."""
    for entry in getattr(item, "prov", None) or []:
        page = getattr(entry, "page_no", 0)
        if page:
            return int(page)
    return 0


def _heading_level(item: Any, label: str, depth: int) -> int:
    """Returns a 1-based heading depth.

    A document title outranks everything, so it is pinned to 1. A section header
    carries its own level when Docling assigns one; otherwise the walk's depth
    stands in, which at least preserves relative nesting.
    """
    if label == "title":
        return 1
    level = getattr(item, "level", None)
    if isinstance(level, int) and level > 0:
        return level + 1
    return max(2, int(depth) + 1)


def _caption_of(item: Any, docling_doc: Any) -> str:
    """Returns an item's caption text, or '' when it has none."""
    getter = getattr(item, "caption_text", None)
    if not callable(getter):
        return ""
    try:
        return " ".join(str(getter(docling_doc) or "").split())
    except Exception:  # noqa: BLE001 - a missing caption is not a failure
        return ""


def _classification_of(item: Any) -> str:
    """Returns the picture classifier's verdict, or '' when it did not run."""
    for annotation in getattr(item, "annotations", None) or []:
        for predicted in getattr(annotation, "predicted_classes", None) or []:
            name = getattr(predicted, "class_name", "")
            if name:
                return str(name)
    return ""


def _table_of(item: Any, docling_doc: Any, page: int) -> Table | None:
    """Builds a `Table` from a Docling table item.

    The grid is preferred over the dataframe export because it is the only form
    that says which cells are *header* cells; a dataframe flattens a two-deep
    header into one row of column names and loses the distinction entirely --
    exactly the information a retrieved financial row needs.
    """
    caption = _caption_of(item, docling_doc)
    grid = getattr(getattr(item, "data", None), "grid", None)
    if grid:
        rows: list[list[str]] = []
        header_rows = 0
        for index, row in enumerate(grid):
            rows.append([" ".join(str(getattr(cell, "text", "") or "").split())
                         for cell in row])
            # Header rows are the leading run only. A "header" cell further down
            # is a spanner inside the body, and treating it as header would put
            # body figures in the column labels.
            if row and index == header_rows and all(
                getattr(cell, "column_header", False) for cell in row
            ):
                header_rows += 1
        rows = [row for row in rows if any(row)]
        if rows:
            return Table(rows=rows, header_rows=min(header_rows, len(rows)),
                         caption=caption, page=page)

    # Fallback: no grid, so take the dataframe and treat its columns as the
    # header. Worse, but a table with an approximate header beats none.
    frame = _dataframe_of(item, docling_doc)
    if frame is None:
        return None
    try:
        header = [" ".join(str(name).split()) for name in frame.columns]
        body = [[" ".join(str(cell).split()) for cell in row] for row in frame.values]
    except Exception:  # noqa: BLE001
        return None
    rows = ([header] if any(header) else []) + body
    if not rows:
        return None
    return Table(rows=rows, header_rows=1 if any(header) else 0,
                 caption=caption, page=page)


def _dataframe_of(item: Any, docling_doc: Any) -> Any:
    """Returns a table item's dataframe export, or None.

    Docling changed this signature between versions, so both are tried.
    """
    for call in (lambda: item.export_to_dataframe(doc=docling_doc),
                 lambda: item.export_to_dataframe()):
        try:
            return call()
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - an unexportable table is skipped
            return None
    return None


def _source_pages(docling_doc: Any, pdf: Path) -> int:
    """Returns the page count the source PDF declares.

    From Docling when it reports one, from the PDF itself otherwise. This is the
    denominator of the coverage check: were it to fall back to the number of
    pages that produced text, the check would always pass.
    """
    pages = getattr(docling_doc, "pages", None)
    if pages:
        try:
            return len(pages)
        except TypeError:
            pass
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except Exception:  # noqa: BLE001 - the coverage check degrades, nothing else
        return 0


def _set_if_present(target: Any, name: str, value: Any) -> None:
    """Sets an attribute only when the target already declares it.

    Docling's pipeline options gain and lose fields between minor versions.
    Setting one blindly on a pydantic model raises, while skipping one that
    exists would silently disable figure extraction -- so presence is tested, and
    an absent field is logged rather than ignored outright.
    """
    if not hasattr(target, name):
        logger.debug("Docling has no pipeline option %r in this version.", name)
        return
    try:
        setattr(target, name, value)
    except Exception as exc:  # noqa: BLE001 - a rejected option is not fatal
        logger.debug("Docling rejected option %s=%r: %s", name, value, exc)


# --- Cache --------------------------------------------------------------------

def write_extraction(document: ExtractedDocument, path: Path) -> None:
    """Writes an extraction to its cache file atomically.

    Atomically because this file is minutes of CPU: a run interrupted midway
    through the write must not leave a truncated JSON that the next run would
    either fail to parse or, worse, parse into a partial document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False), encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def read_extraction(path: Path) -> ExtractedDocument | None:
    """Reads an extraction from its cache file, or None when unusable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Extraction cache unreadable at %s: %s", path, exc)
        return None
    return ExtractedDocument.from_dict(payload)
