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

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
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

# Where `run` records the inputs that determined an extraction, so a later run
# can tell whether the cache on disk is what it would produce. One dict written
# and compared by one pair of methods, rather than a list of fields to keep in
# step: the fast transcript path and the Docling path record different
# diagnostics, and a field-by-field comparison silently never matched for
# transcripts.
CACHE_KEY_FIELD: str = "cache_key"

# Suffix marking an extraction produced by TableFormer's fast mode. Settings
# that change the *text* have to reach `pending_reason`, which compares
# recorded versions and knows nothing about an Extractor instance; folding the
# mode into the version is what makes flipping it re-extract rather than
# silently leave a cache the new setting would not have produced. Device and
# thread count are deliberately absent: the same model on a GPU and on a CPU
# yields the same document, so they must not invalidate anything.
FAST_TABLES_SUFFIX: str = "+fast-tables"

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


def extract_version(accurate_tables: bool = True, page_filter: bool = False) -> str:
    """Returns the extractor version a given set of content settings produces.

    Both arguments change the extracted *text*, so both belong in the version
    Layer 1 compares -- a filtered extraction holds a fraction of the pages an
    unfiltered one does, and reusing one for the other would silently serve a
    partial document. Device and thread count still have no place here: they
    produce the same text, and including them would invalidate every cache the
    moment a run moved machine.

    Args:
        accurate_tables: Whether TableFormer runs in accurate mode.
        page_filter: Whether only the financial section is converted.

    Returns:
        `EXTRACT_VERSION` with a suffix per non-default setting, in a fixed
        order so the same settings always yield the same string.
    """
    from ingestion.documents.sections import PAGE_FILTER_SUFFIX

    version = EXTRACT_VERSION
    if not accurate_tables:
        version += FAST_TABLES_SUFFIX
    if page_filter:
        version += PAGE_FILTER_SUFFIX
    return version


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
        device: Where the layout and table models run -- ``cuda``, ``cpu``,
            ``mps``, ``xpu``. None resolves from `core.hardware`, which is the
            only place that knows whether this torch build can reach the GPU.
        num_threads: CPU threads for model inference and PDF parsing. None
            resolves to the machine's physical core count; Docling's own
            default is four regardless of the machine.
    """

    ocr: bool = True
    figures: bool = True
    accurate_tables: bool = True
    # Convert only an annual report's financial section. Off by default: it
    # drops pages, and a caller has to ask for that explicitly.
    page_filter: bool = False
    cell_matching: bool = True
    image_scale: float = DEFAULT_IMAGE_SCALE
    max_pages: int | None = None
    device: str | None = None
    num_threads: int | None = None
    _converter: Any = field(default=None, init=False, repr=False)
    # What `_apply_accelerator` actually resolved, recorded on every extraction
    # so a slow run can be diagnosed from its cache rather than from memory.
    _resolved_device: str = field(default="", init=False, repr=False)
    _resolved_threads: int = field(default=0, init=False, repr=False)

    # --- Conversion ----------------------------------------------------------

    @property
    def version(self) -> str:
        """Returns the version string this configuration's output carries."""
        return extract_version(self.accurate_tables, self.page_filter)

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
        self._apply_accelerator(options)
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

    def _apply_accelerator(self, options: Any) -> None:
        """Points Docling's models at the resolved device and thread count.

        Left unset, Docling runs four threads on any machine and resolves
        ``device="auto"`` through whichever torch is installed -- which on a
        CPU-only wheel means the GPU is never tried and nothing says so. This
        makes both explicit and logs what was chosen, once per converter.

        A Docling release that renames or drops `AcceleratorOptions` costs the
        acceleration, not the run: the conversion then proceeds on Docling's
        own defaults.
        """
        from core.hardware import DOCLING_DEFAULT_THREADS, profile

        resolved = profile(device=self.device, num_threads=self.num_threads)
        self._resolved_device = resolved.device
        self._resolved_threads = resolved.num_threads
        try:
            from docling.datamodel.pipeline_options import AcceleratorOptions
        except ImportError:
            self._resolved_device = "auto"
            self._resolved_threads = DOCLING_DEFAULT_THREADS
            logger.warning(
                "Docling exposes no AcceleratorOptions; falling back to its "
                "defaults (%d threads, device=auto).",
                DOCLING_DEFAULT_THREADS,
            )
            return

        try:
            options.accelerator_options = AcceleratorOptions(
                device=resolved.device, num_threads=resolved.num_threads
            )
        except Exception as exc:  # noqa: BLE001 - a rejected option is not fatal
            logger.warning("Could not set Docling accelerator options: %s", exc)
            self._resolved_device = "auto"
            self._resolved_threads = DOCLING_DEFAULT_THREADS
            return

        logger.info(
            "Docling running on %s with %d thread(s)%s.",
            resolved.device,
            resolved.num_threads,
            f" [{resolved.gpu_name}]" if resolved.gpu_name else "",
        )
        if resolved.idle_gpu:
            logger.warning(
                "This machine has an NVIDIA GPU that torch %s cannot use "
                "(CPU-only build); extraction will run on the CPU.",
                resolved.torch_build,
            )

    def _cache_key(
        self,
        source: Path,
        limit: int | None,
        source_sha256: str = "",
        page_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Returns the inputs that determine this extraction's content.

        Everything here changes the extracted text; nothing here is merely
        about how fast it was produced. `device` and `num_threads` are
        deliberately absent -- they yield the same document, and including
        them would invalidate every cache the moment a run moved machine,
        which is the same reasoning `extract_version` follows for the table
        mode. `accurate_tables` needs no entry of its own because
        `self.version` already carries it as a suffix.

        Args:
            source: The PDF being extracted.
            limit: Page cap in force for this call.
            source_sha256: Digest supplied by the caller, or empty to hash the
                file here.
            page_range: The pages this call will convert, or None for all of
                them. Two extractions of the same PDF over different ranges are
                different documents, so the range keys the cache.

        Returns:
            A JSON-round-trippable dict, compared whole against what an
            extraction on disk recorded.
        """
        return {
            "extractor": self.version,
            "source_sha256": source_sha256 or _hash_file(source),
            "source_bytes": _file_size(source),
            "max_pages": int(limit) if limit else 0,
            "page_range": list(page_range) if page_range else [],
            "ocr": bool(self.ocr),
            "figures": bool(self.figures),
            "cell_matching": bool(self.cell_matching),
            "image_scale": float(self.image_scale),
        }

    def _cached(
        self, store: DocumentStore, doc_id: str, key: dict[str, Any]
    ) -> ExtractedDocument | None:
        """Returns the extraction on disk when it matches `key`, else None.

        An extraction with no blocks is treated as absent: an empty document
        is how a filing that Docling could not read is recorded, and caching
        that would make one bad pass permanent.
        """
        if not key.get("source_sha256"):
            # The PDF could not be hashed, so nothing can be shown to match it.
            return None
        path = store.extraction(doc_id)
        if not path.exists():
            return None
        document = read_extraction(path)
        if document is None or not document.blocks:
            return None
        if document.meta.get(CACHE_KEY_FIELD) != key:
            return None
        logger.info(
            "[%s] reusing cached extraction: %d block(s) over %d page(s), %s",
            doc_id,
            len(document.blocks),
            document.n_pages,
            document.extractor,
        )
        return document

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
        reuse: bool = True,
        source_sha256: str = "",
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
            reuse: Return the cached extraction when it was produced from this
                same PDF by this same configuration. False forces a real
                Docling pass, which is what a verification harness wants.
            source_sha256: Digest of `pdf`, when the caller already has one --
                `DownloadResult` carries it for both a fresh download and a
                reused file. Left empty, the file is hashed here; a 40 MB
                filing is one read either way, so the point is not to repeat
                it once per run.

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
            raise ExtractionError(f"no such file: {source}")

        limit = max_pages if max_pages is not None else self.max_pages
        started = time.monotonic()

        # Resolved before the cache lookup, not after: the range is part of the
        # key, so a full extraction on disk must not satisfy a filtered request
        # (or the reverse). `financial_page_range` returns None for anything it
        # is not confident about, which means "convert everything".
        page_range: tuple[int, int] | None = None
        if self.page_filter:
            from ingestion.documents.sections import financial_page_range

            page_range = financial_page_range(source)

        key = self._cache_key(source, limit, source_sha256, page_range)

        # A Docling pass over a dense annual report is minutes of CPU, and the
        # result of the last one is already on disk. Reusing it is what makes a
        # run interrupted between extraction and embedding cheap to resume: the
        # batch's Layer 1 knows only that the document was never *embedded*, so
        # it asks for the extraction again either way.
        if reuse:
            cached = self._cached(store, doc_id, key)
            if cached is not None:
                return cached

        # Fast-Path: Transcripts are pure textual dialogue; parse directly in <1s
        is_transcript = (
            doc_type == "concall_transcript"
            or "transcript" in doc_id.lower()
            or "concall" in doc_id.lower()
        )
        if is_transcript:
            try:
                return self._extract_fast_transcript(
                    source=source,
                    store=store,
                    doc_id=doc_id,
                    doc_type=doc_type or "concall_transcript",
                    ticker=ticker,
                    label=label,
                    limit=limit,
                    write=write,
                    started=started,
                    key=key,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Fast-path transcript parser failed (%s); falling back to Docling.",
                    doc_id,
                    exc,
                )

        try:
            convert_kwargs: dict[str, Any] = {
                "max_num_pages": int(limit) if limit else MAX_PAGES,
            }
            if page_range is not None:
                convert_kwargs["page_range"] = page_range
            result = self.converter.convert(str(source), **convert_kwargs)
        except ExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad filing is a failed document
            raise ExtractionError(f"docling could not convert {source}: {exc}") from exc

        docling_doc = getattr(result, "document", None)
        if docling_doc is None:
            raise ExtractionError(f"docling returned no document for {source}")

        document = ExtractedDocument(
            doc_id=doc_id,
            doc_type=doc_type,
            ticker=ticker,
            label=label,
            extractor=self.version,
        )
        document.blocks = list(self._blocks(docling_doc, store, doc_id))
        document.n_source_pages = _source_pages(docling_doc, source)
        document.n_pages = len({block.page for block in document.blocks if block.page})
        document.meta = {
            "page_range": list(page_range) if page_range else [],
            "ocr": bool(self.ocr),
            "figures": bool(self.figures),
            "accurate_tables": bool(self.accurate_tables),
            "cell_matching": bool(self.cell_matching),
            "image_scale": float(self.image_scale),
            "max_pages": int(limit) if limit else 0,
            "device": self._resolved_device,
            "num_threads": self._resolved_threads,
            "seconds": round(time.monotonic() - started, 1),
            CACHE_KEY_FIELD: key,
        }
        self._report(document)

        if write:
            write_extraction(document, store.extraction(doc_id))
        return document

    def _extract_fast_transcript(
        self,
        source: Path,
        store: DocumentStore,
        doc_id: str,
        doc_type: str,
        ticker: str,
        label: str,
        limit: int | None,
        write: bool,
        started: float,
        key: dict[str, Any],
    ) -> ExtractedDocument:
        """High-speed, structural text extractor for concall transcripts (<1s latency)."""
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        total_pages = len(reader.pages)
        max_p = min(total_pages, int(limit)) if limit else total_pages

        document = ExtractedDocument(
            doc_id=doc_id,
            doc_type=doc_type,
            ticker=ticker,
            label=label,
            extractor="fast_transcript/v1",
        )
        blocks: list[Block] = []
        current_section = "Earnings Conference Call"

        for p_idx in range(max_p):
            page_num = p_idx + 1
            page = reader.pages[p_idx]
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            para: list[str] = []
            for line in lines:
                # Filter out header/footer boilerplate lines
                if re.match(
                    r"^(?:Page \d+|\d+ of \d+|Earnings Call|Transcript|BSE Limited|NSE Limited)",
                    line,
                    re.IGNORECASE,
                ):
                    continue

                # Detect Section Boundaries
                if re.search(
                    r"Question.*Answer Session|Q&A Session", line, re.IGNORECASE
                ):
                    if para:
                        blocks.append(
                            Block(
                                kind=KIND_TEXT,
                                text=" ".join(para),
                                page=page_num,
                                path=[current_section],
                            )
                        )
                        para = []
                    current_section = "Question & Answer Session"
                    blocks.append(
                        Block(
                            kind=KIND_HEADING,
                            text=line,
                            page=page_num,
                            level=1,
                            path=[current_section],
                        )
                    )
                    continue

                # Detect speaker turns
                if re.match(
                    r"^(?:[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+|Operator|Moderator|Management|Analyst):",
                    line,
                ):
                    if para:
                        blocks.append(
                            Block(
                                kind=KIND_TEXT,
                                text=" ".join(para),
                                page=page_num,
                                path=[current_section],
                            )
                        )
                        para = []
                    para.append(line)
                else:
                    para.append(line)

            if para:
                blocks.append(
                    Block(
                        kind=KIND_TEXT,
                        text=" ".join(para),
                        page=page_num,
                        path=[current_section],
                    )
                )

        document.blocks = blocks
        document.n_source_pages = total_pages
        document.n_pages = len({b.page for b in blocks if b.page})
        document.meta = {
            "fast_path": True,
            "max_pages": int(limit) if limit else 0,
            "seconds": round(time.monotonic() - started, 3),
            CACHE_KEY_FIELD: key,
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
                    yield Block(
                        kind=KIND_TABLE,
                        text=table.to_markdown(),
                        page=page,
                        path=path,
                        table=table,
                    )
                continue

            if kind == KIND_FIGURE:
                if not self.figures:
                    continue
                figures_on_page[page] += 1
                figure = self._save_figure(
                    item,
                    docling_doc,
                    store,
                    doc_id,
                    page,
                    figures_on_page[page],
                )
                if figure is not None:
                    yield Block(
                        kind=KIND_FIGURE,
                        text=figure.caption,
                        page=page,
                        path=path,
                        figure=figure,
                    )
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
                yield Block(
                    kind=KIND_HEADING,
                    text=text,
                    page=page,
                    level=level,
                    path=[name for _level, name in trail],
                )
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
            logger.debug(
                "[%s] could not render a picture on page %d: %s", doc_id, page, exc
            )
            image = None
        if image is None:
            return None

        width, height = (
            int(getattr(image, "width", 0)),
            int(getattr(image, "height", 0)),
        )
        if width < MIN_FIGURE_PIXELS or height < MIN_FIGURE_PIXELS:
            return None

        target = store.figure(doc_id, page, index)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(target), format="PNG")
        except Exception as exc:  # noqa: BLE001 - as above
            logger.debug(
                "[%s] could not save a picture on page %d: %s", doc_id, page, exc
            )
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
            document.doc_id,
            document.n_pages,
            len(document.blocks),
            document.n_chars,
            ", ".join(f"{pair[0]}={pair[1]}" for pair in sorted(counts.items()))
            or "nothing",
            document.meta.get("seconds", "?"),
        )
        empty = document.empty_pages()
        if empty:
            logger.warning(
                "[%s] %d of %d pages yielded nothing at all (e.g. %s). A scanned "
                "insert needs OCR; without it those pages are silently absent "
                "from the index rather than reported missing.",
                document.doc_id,
                len(empty),
                document.n_source_pages or document.n_pages,
                ", ".join(str(page) for page in empty[:8]),
            )
        text_free = document.text_free_pages()
        if text_free:
            # Ordinary for a deck, notable for an annual report: the figures are
            # on disk but nothing reads them yet.
            logger.info(
                "[%s] %d page(s) produced figures or tables but no text (e.g. %s).",
                document.doc_id,
                len(text_free),
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
            rows.append(
                [" ".join(str(getattr(cell, "text", "") or "").split()) for cell in row]
            )
            # Header rows are the leading run only. A "header" cell further down
            # is a spanner inside the body, and treating it as header would put
            # body figures in the column labels.
            if (
                row
                and index == header_rows
                and all(getattr(cell, "column_header", False) for cell in row)
            ):
                header_rows += 1
        rows = [row for row in rows if any(row)]
        if rows:
            return Table(
                rows=rows,
                header_rows=min(header_rows, len(rows)),
                caption=caption,
                page=page,
            )

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
    return Table(
        rows=rows, header_rows=1 if any(header) else 0, caption=caption, page=page
    )


def _dataframe_of(item: Any, docling_doc: Any) -> Any:
    """Returns a table item's dataframe export, or None.

    Docling changed this signature between versions, so both are tried.
    """
    for call in (
        lambda: item.export_to_dataframe(doc=docling_doc),
        lambda: item.export_to_dataframe(),
    ):
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


def _hash_file(path: Path, block: int = 1 << 20) -> str:
    """Returns a file's sha256, or an empty string when it cannot be read.

    Empty on failure rather than raising: the digest feeds a cache key, and a
    key that cannot be built must mean "extract again", never "crash".
    """
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(block), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    """Returns a file's size in bytes, or 0 when it cannot be stat'd."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def write_extraction(document: ExtractedDocument, path: Path) -> None:
    """Writes an extraction to its cache file atomically.

    Atomically because this file is minutes of CPU: a run interrupted midway
    through the write must not leave a truncated JSON that the next run would
    either fail to parse or, worse, parse into a partial document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_extraction(path: Path) -> ExtractedDocument | None:
    """Reads an extraction from its cache file, or None when unusable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Extraction cache unreadable at %s: %s", path, exc)
        return None
    return ExtractedDocument.from_dict(payload)
