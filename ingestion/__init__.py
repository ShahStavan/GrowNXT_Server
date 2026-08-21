"""Document ingestion layer for GrowNXT.

Turns a company's published filings into embedded, citable evidence:

    catalogue -> download -> parse -> chunk -> embed -> extraction prompt

Every stage records what it consumed in ``data/<TICKER>/registry.json``, so a
second run of the pipeline does nothing the first run already did. See
``ingestion.graph`` for the pipeline and ``ingestion.registry`` for the state
model.
"""

from ingestion.catalog import Catalog, CatalogEntry, CatalogError, fetch_catalog
from ingestion.chunker import Chunk, ChunkSet, chunk_document
from ingestion.embedder import EmbeddingClient, EmbeddingError, load_vectors, save_vectors
from ingestion.fetcher import download_document
from ingestion.graph import IngestionPipeline, IngestionState, ingest
from ingestion.layout import PageText, extract_pdf_pages
from ingestion.parsers import Block, ParsedDocument, Section, parse_document
from ingestion.prompts import (
    DOC_PROFILES,
    REPORT_SECTIONS,
    build_extraction_prompt,
    build_probes,
)
from ingestion.registry import (
    DOC_TYPES,
    STAGES,
    DocumentRecord,
    DocumentRegistry,
)
from ingestion.retrieval import (
    DocumentIndex,
    Hit,
    load_document_index,
    load_ticker_index,
    retrieve_for_probes,
)

__all__ = [
    "Catalog",
    "CatalogEntry",
    "CatalogError",
    "fetch_catalog",
    "download_document",
    "PageText",
    "extract_pdf_pages",
    "Block",
    "Section",
    "ParsedDocument",
    "parse_document",
    "Chunk",
    "ChunkSet",
    "chunk_document",
    "EmbeddingClient",
    "EmbeddingError",
    "save_vectors",
    "load_vectors",
    "DocumentIndex",
    "Hit",
    "load_document_index",
    "load_ticker_index",
    "retrieve_for_probes",
    "DOC_PROFILES",
    "REPORT_SECTIONS",
    "build_probes",
    "build_extraction_prompt",
    "DocumentRegistry",
    "DocumentRecord",
    "DOC_TYPES",
    "STAGES",
    "IngestionPipeline",
    "IngestionState",
    "ingest",
]
