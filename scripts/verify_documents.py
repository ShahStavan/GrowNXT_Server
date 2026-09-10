"""Verification for the document acquisition and extraction layer.

Checks mirror how each module could appear to work while being wrong: an HTML
error page served with a 200, a truncated transfer that still starts with a PDF
header, a 404 retried as transient, a failed attempt leaving a partial file, an
extraction returning something for every page. ``--offline`` skips Docling.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.documents import content as content_model  # noqa: E402
from ingestion.documents.content import (  # noqa: E402
    Block,
    ExtractedDocument,
    Figure,
    Table,
)
from ingestion.documents.download import (  # noqa: E402
    MIN_BYTES,
    Downloader,
    DownloadRequest,
)
from ingestion.documents.storage import DocumentStore  # noqa: E402
from scripts.checks import Report, banner  # noqa: E402

logger = logging.getLogger(__name__)


# --- Fixtures -----------------------------------------------------------------


def _pdf_bytes(pages: int = 3, pad: int = 20000) -> bytes:
    """Builds a valid multi-page PDF in memory.

    Written with pypdf rather than hand-assembled so that the bytes are a PDF a
    real reader accepts -- the point of several checks below is precisely that
    the downloader distinguishes a real PDF from something shaped like one. The
    metadata padding lifts it over MIN_BYTES, since a blank-page PDF is about a
    kilobyte and would be rejected as a stub.

    Args:
        pages: Blank pages to include.
        pad: Characters of filler metadata.

    Returns:
        The PDF's bytes.

    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(max(1, pages)):
        writer.add_blank_page(width=595, height=842)
    if pad:
        writer.add_metadata({"/Producer": "G" * pad})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class _FakeResponse:
    """One scripted HTTP response, shaped like the bits requests exposes."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status_code = status
        self.headers = dict(headers or {})
        self._body = body
        self._error = error

    @property
    def ok(self) -> bool:
        """Mirrors requests' own notion of a successful status."""
        return 200 <= self.status_code < 300

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        """Yields the body in blocks, or raises the scripted error."""
        if self._error is not None:
            raise self._error
        for start in range(0, len(self._body), max(1, chunk_size)):
            yield self._body[start : start + chunk_size]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _StubDownloader(Downloader):
    """A Downloader whose HTTP session is a scripted queue.

    Subclassing rather than monkeypatching keeps the real retry, validation and
    atomic-rename paths under test -- only the socket is replaced.
    """

    def __init__(
        self, store: DocumentStore, responses: Sequence[_FakeResponse], **kwargs: Any
    ) -> None:
        super().__init__(store, **kwargs)
        self.queue: list[_FakeResponse] = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def _session(self) -> _StubDownloader:  # type: ignore[override]
        return self

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,  # noqa: ARG002 - mirrors the real session's signature
    ) -> _FakeResponse:
        """Records the request and returns the next scripted response."""
        self.calls.append({"url": url, "headers": dict(headers or {})})
        if not self.queue:
            raise AssertionError("the stub ran out of scripted responses")
        return self.queue.pop(0)


# --- Storage checks -----------------------------------------------------------


def check_storage(report: Report) -> None:
    """Asserts the on-disk layout is derived consistently."""
    report.section("Storage layout")
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        store = DocumentStore.open("m&m-ltd", data_dir=root)

        report.check(
            "ticker folded for the filesystem",
            store.ticker == "M_M-LTD",
            store.ticker,
        )
        report.check(
            "naming a path creates nothing",
            not store.documents.exists(),
            "documents/ absent until ensure()",
        )

        store.ensure()
        report.check(
            "ensure creates the three directories",
            all(p.is_dir() for p in (store.documents, store.extracted, store.figures)),
            "documents/, extracted/, figures/",
        )

        pdf = store.pdf("annual_report_FY2026")
        extraction = store.extraction("annual_report_FY2026")
        report.check(
            "pdf lands under documents/",
            pdf.parent == store.documents and pdf.name == "annual_report_FY2026.pdf",
            store.relative(pdf),
        )
        report.check(
            "extraction lands under extracted/",
            extraction.parent == store.extracted
            and extraction.name == "annual_report_FY2026.json",
            store.relative(extraction),
        )

        # Zero padding is the whole point: a listing must sort into page order,
        # which naive numbering breaks at page 10 and again at page 100.
        names = [store.figure("d", page, 1).name for page in (2, 10, 142)]
        report.check(
            "figure names sort in page order",
            names == sorted(names),
            ", ".join(names),
        )
        report.check(
            "figure path is per document",
            store.figure("d", 1, 1).parent == store.figures / "d",
            store.relative(store.figure("d", 1, 1)),
        )

        relative = store.relative(pdf)
        report.check(
            "relative and resolve round-trip",
            store.resolve(relative).resolve() == pdf.resolve(),
            relative,
        )
        outside = root / "elsewhere" / "other.pdf"
        report.check(
            "a path outside the store is not forced",
            store.relative(outside).endswith("elsewhere/other.pdf"),
            store.relative(outside),
        )


# --- Content model checks -----------------------------------------------------


def check_content(report: Report) -> None:
    """Asserts the extracted-content model preserves what retrieval needs."""
    report.section("Content model")

    table = Table(
        rows=[
            ["Particulars", "FY2026", "FY2025"],
            ["Rs in millions", "", ""],
            ["Revenue from operations", "926,163", "790,935"],
            ["Profit for the year", "112,004", "104,180"],
        ],
        header_rows=2,
        caption="Consolidated Statement of Profit and Loss",
        page=142,
    )

    report.check(
        "multi-row header merges column-wise",
        table.header == ["Particulars Rs in millions", "FY2026", "FY2025"],
        " / ".join(table.header),
    )
    report.check(
        "body excludes the header rows",
        len(table.body) == 2 and table.body[0][0] == "Revenue from operations",
        f"{len(table.body)} body rows",
    )

    markdown = table.to_markdown()
    report.check(
        "markdown carries the caption",
        markdown.startswith("Consolidated Statement of Profit and Loss"),
        "caption present, so the scale travels with the figures",
    )
    report.check(
        "markdown carries the header row",
        "FY2026" in markdown.split("\n")[1],
        "columns are labelled in the rendering",
    )

    piped = Table(rows=[["a|b", "c"]], header_rows=1)
    report.check(
        "a pipe in a cell is escaped",
        "a\\|b" in piped.to_markdown(),
        "rendering stays valid markdown",
    )

    headerless = Table(rows=[["1", "2", "3"]], header_rows=0)
    report.check(
        "a headerless table is recorded as such",
        headerless.header == [] and headerless.to_markdown() != "",
        "absence of column labels is visible, not papered over",
    )

    # The heading trail is what phase-2 sectioning is built on.
    document = ExtractedDocument(
        doc_id="annual_report_FY2026",
        doc_type="annual_report",
        ticker="WIPRO",
        label="FY2026",
        n_pages=3,
        n_source_pages=5,
        extractor="test",
        blocks=[
            Block(
                kind=content_model.KIND_HEADING,
                text="Management Discussion",
                page=1,
                level=1,
                path=["Management Discussion"],
            ),
            Block(
                kind=content_model.KIND_TEXT,
                text="Revenue grew on large deals.",
                page=1,
                path=["Management Discussion", "Segment Review"],
            ),
            Block(
                kind=content_model.KIND_TABLE,
                text=markdown,
                page=2,
                path=["Management Discussion"],
                table=table,
            ),
            Block(
                kind=content_model.KIND_FIGURE,
                text="",
                page=3,
                path=["Management Discussion"],
                figure=Figure(
                    path="figures/d/p0003-01.png",
                    kind="bar_chart",
                    page=3,
                    width=800,
                    height=600,
                ),
            ),
        ],
    )

    report.check(
        "a block knows the headings it sits under",
        document.blocks[1].heading_trail == "Management Discussion > Segment Review",
        document.blocks[1].heading_trail,
    )
    report.check(
        "sections come from the document's own headings",
        document.sections() == ["Management Discussion"],
        ", ".join(document.sections()),
    )
    report.check(
        "tables and figures are addressable",
        len(document.tables) == 1 and len(document.figures) == 1,
        "1 table, 1 figure",
    )

    # Pages 4 and 5 exist in the source but produced nothing. That has to be
    # visible: a silent gap is indistinguishable from a document that is short.
    report.check(
        "pages that produced nothing are named",
        document.empty_pages() == [4, 5],
        f"empty pages: {document.empty_pages()}",
    )
    # Page 3 holds a chart and no prose. That is a successful extraction of a
    # deck page, not a gap, so it must not be counted as empty.
    report.check(
        "a figure-only page counts as covered",
        3 not in document.empty_pages() and document.text_free_pages() == [3],
        f"text-free but covered: {document.text_free_pages()}",
    )

    restored = ExtractedDocument.from_dict(document.to_dict())
    report.check(
        "the document round-trips through JSON",
        (
            restored.to_dict() == document.to_dict()
            and restored.blocks[2].table is not None
            and restored.blocks[2].table.header_rows == 2
        ),
        f"{len(restored.blocks)} blocks, header rows preserved",
    )
    report.check(
        "markdown rendering is readable",
        "# Management Discussion" in document.to_markdown(),
        "headings render as markdown headings",
    )


# --- Download checks ----------------------------------------------------------


def check_download(report: Report) -> None:
    """Asserts what does and does not reach the documents directory."""
    report.section("Acquisition")
    pdf = _pdf_bytes(pages=3)
    url = "https://nsearchives.example.com/corporate/AR_2026.pdf"

    def store_in(root: Path) -> DocumentStore:
        return DocumentStore.open("WIPRO", data_dir=root).ensure()

    # A good download.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        stub = _StubDownloader(store, [_FakeResponse(200, pdf)])
        result = stub.fetch(DownloadRequest("annual_report_FY2026", url, "FY2026"))
        report.check(
            "a valid PDF is stored",
            result.ok and result.path is not None and result.path.exists(),
            "%.1f KB on disk" % (result.n_bytes / 1024.0),
        )
        report.check(
            "the hash is of the served bytes",
            result.sha256 == __import__("hashlib").sha256(pdf).hexdigest(),
            result.sha256[:16] + "...",
        )
        report.check(
            "the page count is read back",
            result.n_pages == 3,
            f"{result.n_pages} pages",
        )
        report.check(
            "a same-origin referer is sent",
            stub.calls[0]["headers"].get("Referer")
            == "https://nsearchives.example.com/",
            stub.calls[0]["headers"].get("Referer", "(none)"),
        )
        report.check(
            "a browser user agent is sent",
            "Chrome" in stub.calls[0]["headers"].get("User-Agent", ""),
            "exchange hosts refuse anything else",
        )

        # Second fetch, no scripted responses left: reuse or bust.
        again = _StubDownloader(store, [])
        reused = again.fetch(DownloadRequest("annual_report_FY2026", url))
        report.check(
            "an existing valid file is reused, not re-fetched",
            reused.ok and reused.reused and not again.calls,
            "no HTTP request made",
        )

        forced = _StubDownloader(store, [_FakeResponse(200, pdf)])
        refetched = forced.fetch(
            DownloadRequest("annual_report_FY2026", url), force=True
        )
        report.check(
            "force re-downloads regardless",
            refetched.ok and not refetched.reused and len(forced.calls) == 1,
            "1 HTTP request made",
        )

    # An HTML error page served with a 200 -- the common IR-host failure.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        html = b"<html><body>Document not found</body></html>" * 300
        stub = _StubDownloader(
            store, [_FakeResponse(200, html, {"Content-Type": "text/html"})]
        )
        result = stub.fetch(DownloadRequest("transcript_2026_07", url))
        report.check(
            "a 200 carrying HTML is refused",
            not result.ok and "not a PDF" in result.error,
            result.error[:60],
        )
        report.check(
            "the refused response is not retried",
            result.attempts == 1 and len(stub.calls) == 1,
            "1 attempt: the same bytes would come back",
        )
        report.check(
            "nothing is left in documents/",
            not any(store.documents.iterdir()),
            "no file, no .part",
        )

    # A truncated transfer: begins with %PDF, has no readable page tree.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        truncated = pdf[:9000]
        stub = _StubDownloader(store, [_FakeResponse(200, truncated)], retries=0)
        result = stub.fetch(DownloadRequest("annual_report_FY2026", url))
        report.check(
            "a truncated PDF is refused despite the header",
            not result.ok and "readable pages" in result.error,
            result.error[:60],
        )
        report.check(
            "no partial file survives the failure",
            list(store.documents.glob("*")) == [],
            "documents/ is empty",
        )

    # Too small to be a filing.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        stub = _StubDownloader(store, [_FakeResponse(200, b"%PDF-1.4 tiny")], retries=0)
        result = stub.fetch(DownloadRequest("presentation_2026_07", url))
        report.check(
            "a stub response is refused",
            not result.ok and "too small" in result.error,
            f"floor is {MIN_BYTES} bytes",
        )

    # Permanent versus transient status handling.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        stub = _StubDownloader(store, [_FakeResponse(404)], retries=2)
        result = stub.fetch(DownloadRequest("annual_report_FY2025", url))
        report.check(
            "a 404 is not retried",
            not result.ok and result.attempts == 1 and len(stub.calls) == 1,
            f"{result.attempts} attempt(s) for HTTP 404",
        )

    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        stub = _StubDownloader(
            store,
            [_FakeResponse(503, headers={"Retry-After": "0"}), _FakeResponse(200, pdf)],
            retries=2,
        )
        result = stub.fetch(DownloadRequest("annual_report_FY2025", url))
        report.check(
            "a 503 is retried and can succeed",
            result.ok and result.attempts == 2,
            f"{result.attempts} attempts",
        )

    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        oversized = {"Content-Length": str(500 * 1024 * 1024)}
        stub = _StubDownloader(store, [_FakeResponse(200, pdf, oversized)], retries=2)
        result = stub.fetch(DownloadRequest("annual_report_FY2024", url))
        report.check(
            "a declared size over the ceiling is refused up front",
            not result.ok and "ceiling" in result.error and result.attempts == 1,
            result.error[:60],
        )

    # Concurrency: every request gets exactly one result, failures included.
    with tempfile.TemporaryDirectory() as raw:
        store = store_in(Path(raw))
        stub = _StubDownloader(
            store,
            [_FakeResponse(200, pdf), _FakeResponse(404), _FakeResponse(200, pdf)],
            retries=0,
            workers=1,
        )
        wanted = [
            DownloadRequest("doc_a", url + "?a"),
            DownloadRequest("doc_b", url + "?b"),
            DownloadRequest("doc_c", url + "?c"),
        ]
        results = stub.start(wanted).wait()
        report.check(
            "every request yields exactly one result",
            len(results) == len(wanted),
            f"{len(results)} results for {len(wanted)} requests",
        )
        report.check(
            "a failure among successes does not abort the batch",
            sum(1 for r in results if r.ok) == 2
            and sum(1 for r in results if not r.ok) == 1,
            "2 stored, 1 failed",
        )
        report.check(
            "an empty batch is not an error",
            Downloader(store).start([]).wait() == [],
            "no workers started",
        )


# --- Extraction checks --------------------------------------------------------


def check_extraction(report: Report) -> None:
    """Asserts Docling extraction against a generated PDF.

    Separated from the offline checks because it needs the converter and its
    model weights. What is asserted is structure, not volume: an extraction that
    returns something for every page can still have lost every table.
    """
    report.section("Extraction")
    try:
        from ingestion.documents.extract import EXTRACT_VERSION, Extractor
    except ImportError as exc:
        report.check("docling is importable", False, str(exc)[:70])
        return

    report.check("docling is importable", True, EXTRACT_VERSION)

    fixture = _table_pdf()
    if fixture is None:
        report.check("fixture PDF built", False, "could not build a fixture")
        return

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        store = DocumentStore.open("TESTCO", data_dir=root).ensure()
        source = root / "fixture.pdf"
        source.write_bytes(fixture)

        extractor = Extractor(ocr=False, figures=True)
        document = extractor.run(
            source,
            store,
            doc_id="fixture",
            doc_type="annual_report",
            ticker="TESTCO",
            label="FY2026",
            # The harness asserts what Docling produces, so it must never be
            # handed a cached extraction from an earlier check.
            reuse=False,
        )

        report.check(
            "text is recovered",
            document.n_chars > 0,
            f"{document.n_chars} chars across {len(document.blocks)} blocks",
        )
        report.check(
            "the source page count is recorded",
            document.n_source_pages >= 1,
            f"{document.n_source_pages} source pages",
        )
        report.check(
            "a heading was found",
            len(document.headings) >= 1,
            f"{len(document.headings)} heading(s): "
            f"{', '.join(h.text[:24] for h in document.headings[:3])}",
        )
        report.check(
            "blocks carry a heading trail",
            any(block.path for block in document.blocks),
            "deepest trail: "
            f"{max((block.heading_trail for block in document.blocks), key=len, default='')[:60]}",
        )
        tables = document.tables
        report.check(
            "a table is recovered as a table",
            len(tables) >= 1,
            f"{len(tables)} table(s)",
        )
        if tables:
            first = tables[0]
            report.check(
                "the table keeps its header row",
                first.header_rows >= 1 and any("FY" in cell for cell in first.header),
                f"header: {' / '.join(first.header)[:60]}",
            )
            report.check(
                "the table keeps its figures",
                any(
                    "926,163" in cell or "926163" in cell
                    for row in first.rows
                    for cell in row
                ),
                "row values survived extraction",
            )
        report.check(
            "the extraction round-trips through its cache",
            ExtractedDocument.from_dict(document.to_dict()).to_dict()
            == document.to_dict(),
            store.relative(store.extraction("fixture")),
        )
        report.check(
            "the extraction is written to the store",
            store.extraction("fixture").exists(),
            store.relative(store.extraction("fixture")),
        )


def _table_pdf() -> bytes | None:
    """Renders a one-page PDF holding a heading and a small financial table.

    Built with matplotlib because it is already a dependency and can lay out
    real text into a real PDF; the alternative -- checking a binary fixture into
    the repository -- hides what is being tested.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return None

    rows = [
        ("Particulars", "FY2026", "FY2025"),
        ("Revenue from operations", "926,163", "790,935"),
        ("Profit for the year", "112,004", "104,180"),
    ]
    buffer = io.BytesIO()
    with PdfPages(buffer) as pages:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.suptitle(
            "Management Discussion and Analysis", fontsize=24, fontweight="bold", y=0.96
        )
        figure.text(
            0.1, 0.90, "Consolidated results for the year, Rs in millions.", fontsize=12
        )
        table = figure.add_axes([0.1, 0.55, 0.8, 0.25])
        table.axis("off")
        rendered = table.table(
            cellText=[list(r) for r in rows[1:]], colLabels=list(rows[0]), loc="center"
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(11)
        pages.savefig(figure)
        plt.close(figure)
    return buffer.getvalue()


def check_page_filter(report: Report) -> None:
    """Guards the page-filter anchors and its fail-open rule.

    Restores the `check_page_filter` guard that CLAUDE.md credits and that was
    deleted with `verify_nifty50_embeddings.py`. The anchor set is the whole
    point: page 191 of the ADANIENT FY2026 filing opens "INDEPENDENT AUDITOR'S
    CERTIFICATE ON COMPLIANCE WITH THE CORPORATE GOVERNANCE REQUIREMENTS", and
    an anchor matching bare "independent auditor" fires there, dragging in 31
    pages of the governance report the filter exists to drop.
    """
    from ingestion.documents import sections

    report.section("Page filter")

    certificate = (
        "INDEPENDENT AUDITOR'S CERTIFICATE ON COMPLIANCE WITH THE "
        "CORPORATE GOVERNANCE REQUIREMENTS"
    )
    report.check(
        "governance certificate does not anchor",
        not any(p.search(certificate) for p in sections._ANCHOR_RE),
        "an anchor matching bare 'independent auditor' would fire here",
    )
    for heading in (
        "INDEPENDENT AUDITOR'S REPORT",
        "INDEPENDENT AUDITORS' REPORT",
        "Management Discussion and Analysis",
        "Notes forming part of the financial statements",
        "Consolidated Balance Sheet",
    ):
        report.check(
            f"anchors on {heading[:38]!r}",
            any(p.search(heading) for p in sections._ANCHOR_RE),
        )

    report.check(
        "PAGE_FILTER_SUFFIX is '+fin-pages'",
        sections.PAGE_FILTER_SUFFIX == "+fin-pages",
        "folded into the extraction cache key",
    )
    report.check(
        "a short document is never filtered",
        sections.financial_page_range(__file__, min_pages=60) is None,
        "fails toward keeping pages",
    )
    report.check(
        "an unreadable file yields no opinion",
        sections.page_texts(Path(__file__).parent / "does-not-exist.pdf") == [],
    )


# --- Entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Runs the suite and returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Verify the document acquisition and extraction layer."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the checks that need Docling and its weights.",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    banner("Document layer verification")
    report = Report()
    check_storage(report)
    check_content(report)
    check_download(report)
    check_page_filter(report)
    if not args.offline:
        check_extraction(report)
    else:
        report.section("Extraction")
        print("  (skipped: --offline)")

    print(report.render())
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
