"""Document ingestion & Qualitative RAG layer for GrowNXT.

Turns a company's published filings into extracted, multimodal citable evidence
and institutional-grade equity research findings:

    catalog -> download -> extract -> chunk -> embed (Qdrant + Snowflake-Arctic) -> RAG (LangGraph)

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from ingestion.catalog import Catalog, CatalogEntry, CatalogError, fetch_catalog
from ingestion.chunker import (
    CHUNKER_VERSION,
    ELEMENT_FIGURE,
    ELEMENT_TABLE,
    ELEMENT_TEXT,
    ELEMENT_TYPES,
    Chunk,
    ChunkSet,
    chunk_document,
    read_chunk_cache,
    write_chunk_cache,
)
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
    DOC_TYPES,
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
    STAGES,
    DocumentRecord,
    FetchResult,
    StageState,
    download_document,
    sha256_file,
    sha256_text,
    utc_now,
)
from ingestion.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_VECTOR_SIZE,
    DocumentIndexState,
    IndexerConfig,
    IndexResult,
    QdrantVectorIndexer,
    TickerState,
    index_ticker_documents,
)
from ingestion.rag import (
    EvidenceChunk,
    EvidenceReranker,
    InstitutionalRAGPipeline,
    InstitutionalSynthesizer,
    ParallelVectorRetriever,
    ResearchDossier,
    ThematicFinding,
    ThematicProbe,
    build_adaptive_probes,
    extract_ticker_findings,
    get_default_probes,
)

__all__ = [
    "Catalog",
    "CatalogEntry",
    "CatalogError",
    "fetch_catalog",
    "download_document",
    "DocumentStore",
    "Downloader",
    "DownloadRequest",
    "Extractor",
    "read_extraction",
    "write_extraction",
    "Block",
    "Table",
    "Figure",
    "ExtractedDocument",
    "Chunk",
    "ChunkSet",
    "chunk_document",
    "read_chunk_cache",
    "write_chunk_cache",
    "CHUNKER_VERSION",
    "ELEMENT_TEXT",
    "ELEMENT_TABLE",
    "ELEMENT_FIGURE",
    "ELEMENT_TYPES",
    "DocumentRecord",
    "StageState",
    "FetchResult",
    "DOC_TYPES",
    "DOC_TYPE_ANNUAL_REPORT",
    "DOC_TYPE_TRANSCRIPT",
    "DOC_TYPE_PRESENTATION",
    "STAGES",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL_ALIAS",
    "DEFAULT_VECTOR_SIZE",
    "DocumentIndexState",
    "IndexerConfig",
    "IndexResult",
    "QdrantVectorIndexer",
    "TickerState",
    "index_ticker_documents",
    "ThematicProbe",
    "EvidenceChunk",
    "ThematicFinding",
    "ResearchDossier",
    "ParallelVectorRetriever",
    "EvidenceReranker",
    "InstitutionalSynthesizer",
    "InstitutionalRAGPipeline",
    "extract_ticker_findings",
    "build_adaptive_probes",
    "get_default_probes",
    "sha256_file",
    "sha256_text",
    "utc_now",
]
