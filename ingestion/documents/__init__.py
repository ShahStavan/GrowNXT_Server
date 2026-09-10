"""Document acquisition and extraction: catalogue URL to parsed filing on disk.

`storage` owns every path, `download` fetches and validates PDFs, `extract`
converts them with Docling into `content.ExtractedDocument`. What a chunk is,
and what a section means, is decided downstream.
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
_LAZY = frozenset(
    {
        "EXTRACT_VERSION",
        "ExtractionError",
        "Extractor",
        "read_extraction",
        "write_extraction",
    }
)

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
    """Imports the extraction module lazily, on first attribute access."""
    if name in _LAZY:
        from ingestion.documents import extract

        return getattr(extract, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
