"""Document acquisition and extraction: getting a filing off the web and read.

This package owns the first half of ingestion -- everything up to the point where
a filing is text, tables and figures on disk. What a *chunk* is, and which
section a passage belongs to, is decided downstream; nothing here has an opinion
about it.

Three modules, one job each:

* `storage` -- every path this package reads or writes, so "where did that
  document go?" has one answer.
* `download` -- concurrent, resumable acquisition of catalogued PDFs. Validates
  that what arrived is a filing rather than an error page, and never leaves a
  partial file behind.
* `extract` -- Docling conversion into `content.ExtractedDocument`: reading order
  from a layout model, table structure from TableFormer, figures as PNGs, and
  OCR for pages that are pictures of text.

`content` holds the data model the last of those fills and the chunker consumes.

Where everything lands, under ``output/<TICKER>/``::

    documents/<doc_id>.pdf        the filing as published, never rewritten
    extracted/<doc_id>.json       text, tables, and the figure manifest
    figures/<doc_id>/p0142-03.png images cropped from the pages

A minimal ingest of one company's filings::

    from ingestion.documents import DocumentStore, Downloader, DownloadRequest
    from ingestion.documents import Extractor

    store = DocumentStore.open("WIPRO").ensure()
    wanted = [DownloadRequest("annual_report_FY2026", url, "FY2026")]

    # Downloads run on a thread pool; results arrive as they land.
    batch = Downloader(store).start(wanted)
    extractor = Extractor()
    for result in batch.completed():
        if result.ok:
            extractor.run(result.path, store, result.doc_id,
                          doc_type="annual_report", ticker="WIPRO", label="FY2026")

Google Python Style Guide Compliant.
"""

from ingestion.documents.content import (
    Block,
    ExtractedDocument,
    Figure,
    Table,
)
from ingestion.documents.download import (
    DOWNLOAD_VERSION,
    DownloadBatch,
    Downloader,
    DownloadRequest,
    DownloadResult,
)
from ingestion.documents.storage import DocumentStore

# `extract` pulls in Docling, which loads torch. Importing it eagerly would make
# every consumer of the package -- the downloader included -- pay seconds of
# import time for models they may never use, so it is imported on demand.
_LAZY = frozenset({
    "EXTRACT_VERSION",
    "ExtractionError",
    "Extractor",
    "read_extraction",
    "write_extraction",
})

__all__ = [
    "Block",
    "DOWNLOAD_VERSION",
    "DocumentStore",
    "DownloadBatch",
    "DownloadRequest",
    "DownloadResult",
    "Downloader",
    "ExtractedDocument",
    "Figure",
    "Table",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> object:
    """Imports the extraction module lazily, on first attribute access.

    Keeps `import ingestion.documents` cheap while still letting a caller write
    ``from ingestion.documents import Extractor``.
    """
    if name in _LAZY:
        from ingestion.documents import extract
        return getattr(extract, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
