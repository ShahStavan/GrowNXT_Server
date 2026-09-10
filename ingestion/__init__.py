"""Document acquisition layer: catalogue, download, Docling extraction.

Chunking, Arctic/Qdrant embedding, the Nifty 50 batch orchestrator and the
LangGraph RAG pipeline were removed ahead of the vectorless qualitative
rebuild; recover any of it from the commit before that. What survives is
the half that was never about vectors, and the rebuild builds on it.
"""

from __future__ import annotations

from ingestion.catalog import Catalog, CatalogEntry, CatalogError, fetch_catalog
from ingestion.documents.content import (
    Block,
    ExtractedDocument,
    Figure,
    Table,
)
from ingestion.documents.download import (
    Downloader,
    DownloadRequest,
)
from ingestion.documents.extract import (
    Extractor,
    read_extraction,
    write_extraction,
)
from ingestion.documents.storage import (
    DocumentStore,
)
from ingestion.fetcher import (
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
    DOC_TYPES,
    STAGES,
    DocumentRecord,
    FetchResult,
    StageState,
    download_document,
    sha256_file,
    sha256_text,
    utc_now,
)

__all__ = [
    # Catalogue
    "Catalog",
    "CatalogEntry",
    "CatalogError",
    "fetch_catalog",
    # Document content model
    "Block",
    "ExtractedDocument",
    "Figure",
    "Table",
    # Acquisition
    "Downloader",
    "DownloadRequest",
    "DocumentStore",
    # Extraction
    "Extractor",
    "read_extraction",
    "write_extraction",
    # Fetcher contracts
    "DOC_TYPE_ANNUAL_REPORT",
    "DOC_TYPE_PRESENTATION",
    "DOC_TYPE_TRANSCRIPT",
    "DOC_TYPES",
    "STAGES",
    "DocumentRecord",
    "FetchResult",
    "StageState",
    "download_document",
    "sha256_file",
    "sha256_text",
    "utc_now",
]
