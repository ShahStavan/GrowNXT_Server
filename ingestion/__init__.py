"""Document acquisition layer for GrowNXT.

Turns a company's published filings into extracted, citable document content:

    catalog -> download -> extract (Docling)

**What used to be here.** This package also carried chunking, Snowflake Arctic
embedding into Qdrant, the Nifty 50 batch orchestrator and a LangGraph RAG
pipeline. All of it was removed ahead of the vectorless qualitative rebuild
described in `.claude/specs/vectorless-qualitative-rag.md`; the replacement is
a section map and a typed evidence ledger, not a vector index. The removal
commit is the place to recover any of it from.

What survives is the half that was never about vectors: fetching a document
catalogue, downloading filings, and converting them with Docling. The new
qualitative pipeline builds directly on these.

Google Python Style Guide Compliant.
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
