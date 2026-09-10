"""Qdrant vector indexing and incremental state tracking engine for GrowNXT.

Embeds multimodal corporate filing chunks (prose text, structured tables, and
figures) with `Snowflake/snowflake-arctic-embed-m-v1.5` into 768-dimensional
dense vectors and stores them in Qdrant with rich payload metadata,
deterministic point IDs, and incremental change detection via per-ticker state
tracking.

Each stock's vectors live in their own collection by default
(``<prefix>_<ticker>``, see `IndexerConfig.collection_for`), so a stock can be
dropped, rebuilt or searched without touching -- or filtering across -- any
other. The per-stock mirror written by `write_local_vector_cache`
(``output/<TICKER>/vectors.npz`` + ``payloads.json``) makes the same
embeddings portable and is what the retriever's Tier-1 matrix search reads.

Follows the CocoIndex PDF Elements architecture.
Google Python Style Guide Compliant.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dotenv
from qdrant_client import QdrantClient, models

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from core.config import OUTPUT_DIR, safe_ticker
from ingestion.chunker import (
    ELEMENT_FIGURE,
    ELEMENT_TABLE,
    ELEMENT_TEXT,
    Chunk,
    ChunkSet,
    read_chunk_cache,
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
dotenv.load_dotenv()

DEFAULT_EMBEDDING_MODEL: str = "Snowflake/snowflake-arctic-embed-m-v1.5"
DEFAULT_MODEL_ALIAS: str = "snowflake-arctic-embed-m-v1.5"
DEFAULT_COLLECTION_NAME: str = "grownxt_financial_elements"
# Per-ticker collections are named ``<prefix>_<safe_ticker>``, lower-cased.
DEFAULT_COLLECTION_PREFIX: str = "grownxt"
DEFAULT_VECTOR_SIZE: int = 768
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_UPSERT_BATCH_SIZE: int = 128
# Qdrant's gRPC transport carries vectors as binary rather than JSON text.
# A bulk upsert of a few thousand 768-float rows is roughly a third of the
# bytes and none of the float-to-decimal encoding, so it is the default. A
# cluster that does not expose the gRPC port falls back to REST at connect
# time rather than failing: the transport is a performance choice, and a run
# must never die because one end of it is unavailable.
DEFAULT_PREFER_GRPC: bool = True
STATE_FILENAME: str = "state.json"

# Per-ticker local vector cache read by the retriever's Tier-1 matrix search.
LOCAL_VECTORS_FILENAME: str = "vectors.npz"
LOCAL_PAYLOADS_FILENAME: str = "payloads.json"
# Payload fields the retriever reads back; everything else stays in Qdrant only.
LOCAL_PAYLOAD_FIELDS: tuple[str, ...] = (
    "ticker",
    "doc_id",
    "doc_type",
    "label",
    "chunk_id",
    "element_type",
    "page_start",
    "page_end",
    "section_breadcrumb",
    "content",
    "metadata",
)


def _env_flag(name: str, default: bool) -> bool:
    """Reads a boolean environment variable (``1/true/yes/on``)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Fixed UUIDv5 namespace for GrowNXT deterministic point IDs
NAMESPACE_GROWNXT: uuid.UUID = uuid.UUID("a74f85e2-2b63-4c6e-9e77-cf867d712f5a")


def utc_now() -> str:
    """Returns the current UTC timestamp formatted as ISO-8601 string."""
    return _datetime.datetime.now(_datetime.UTC).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    """Returns the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_uuid(ticker: str, doc_id: str, chunk_id: str) -> str:
    """Generates a deterministic UUIDv5 string for a given chunk."""
    seed = f"{ticker}_{doc_id}_{chunk_id}"
    return str(uuid.uuid5(NAMESPACE_GROWNXT, seed))


@dataclass
class IndexerConfig:
    """Configuration for vector embedding and Qdrant indexing."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    model_alias: str = DEFAULT_MODEL_ALIAS
    collection_name: str = DEFAULT_COLLECTION_NAME
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX
    # One collection per ticker (``grownxt_tcs``) so every stock's vectors are
    # independent: droppable, rebuildable and searchable without a filter.
    collection_per_ticker: bool = True
    vector_size: int = DEFAULT_VECTOR_SIZE
    distance: str = "Cosine"
    batch_size: int = DEFAULT_BATCH_SIZE
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    # Binary transport for bulk upserts, with a REST fallback proven at
    # connect time. QDRANT_PREFER_GRPC=0 opts out.
    prefer_grpc: bool = DEFAULT_PREFER_GRPC
    # None lets SentenceTransformers choose, which is right only when torch can
    # see the accelerator; `from_env` resolves it through `core.hardware` so a
    # CPU-only torch build on a GPU machine is reported rather than obeyed.
    device: str | None = None
    # Half precision on the embedding model: a straight throughput win on any
    # CUDA card and a slowdown on CPU, so it follows the device.
    fp16: bool = False
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    state_filename: str = STATE_FILENAME

    def collection_for(self, ticker: str | None) -> str:
        """Returns the collection a ticker's vectors live in.

        Falls back to the shared ``collection_name`` when per-ticker mode is
        off or no ticker is known (a cross-ticker query).
        """
        if not self.collection_per_ticker or not ticker:
            return self.collection_name
        return f"{self.collection_prefix}_{safe_ticker(ticker)}".lower()

    @classmethod
    def from_env(cls, **overrides: Any) -> IndexerConfig:
        """Creates config loading connection parameters from environment variables."""
        qdrant_url = (
            overrides.get("qdrant_url")
            or os.getenv("QDRANT_API_URL")
            or os.getenv("QDRANT_URL")
            or os.getenv("QDRANT_CLUSTER")
            or os.getenv("QDRANT_HOST")
        )
        qdrant_api_key = overrides.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY")
        collection_name = (
            overrides.get("collection_name")
            or os.getenv("QDRANT_COLLECTION_NAME")
            or os.getenv("QDRANT_COLLECTION")
            or DEFAULT_COLLECTION_NAME
        )
        # Per-ticker collections are named after the configured collection so
        # ``QDRANT_COLLECTION_NAME=grownxt-embeddings`` yields
        # ``grownxt-embeddings_tcs`` rather than an unrelated prefix.
        collection_prefix = (
            overrides.get("collection_prefix")
            or os.getenv("QDRANT_COLLECTION_PREFIX")
            or os.getenv("QDRANT_COLLECTION_NAME")
            or DEFAULT_COLLECTION_PREFIX
        )
        collection_per_ticker = overrides.get("collection_per_ticker")
        if collection_per_ticker is None:
            collection_per_ticker = _env_flag("QDRANT_COLLECTION_PER_TICKER", True)
        prefer_grpc = overrides.get("prefer_grpc")
        if prefer_grpc is None:
            prefer_grpc = _env_flag("QDRANT_PREFER_GRPC", DEFAULT_PREFER_GRPC)
        model_name = (
            overrides.get("model_name")
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        model_alias = overrides.get("model_alias") or model_name.split("/")[-1]

        # Device, batch size and precision are one decision about one machine,
        # so they come from one place rather than three environment variables
        # read here. An explicit override still wins: `profile` takes it first.
        from core.hardware import profile

        hardware = profile(
            device=overrides.get("device"),
            embed_batch_size=overrides.get("batch_size"),
            embed_fp16=overrides.get("fp16"),
        )

        consumed = (
            "qdrant_url",
            "qdrant_api_key",
            "collection_name",
            "collection_prefix",
            "collection_per_ticker",
            "prefer_grpc",
            "model_name",
            "model_alias",
            "device",
            "batch_size",
            "fp16",
        )
        return cls(
            model_name=model_name,
            model_alias=model_alias,
            collection_name=collection_name,
            collection_prefix=collection_prefix,
            collection_per_ticker=bool(collection_per_ticker),
            prefer_grpc=bool(prefer_grpc),
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            device=hardware.device,
            batch_size=hardware.embed_batch_size,
            fp16=hardware.embed_fp16,
            **{k: v for k, v in overrides.items() if k not in consumed},
        )


@dataclass
class DocumentIndexState:
    """State record for an indexed filing document."""

    fingerprint: str
    chunk_count: int
    embedding_model: str
    qdrant_collection: str
    indexed_at: str
    status: str = "INDEXED"
    # Pipeline versions the chunk file was produced with. A bump to either
    # means the cached chunks describe text the pipeline would no longer
    # produce, so the batch job re-extracts rather than trusting the cache.
    chunker_version: str = ""
    extract_version: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentIndexState:
        return cls(
            fingerprint=str(data.get("fingerprint", "")),
            chunk_count=int(data.get("chunk_count", 0)),
            embedding_model=str(data.get("embedding_model", "")),
            qdrant_collection=str(data.get("qdrant_collection", "")),
            indexed_at=str(data.get("indexed_at", "")),
            status=str(data.get("status", "INDEXED")),
            chunker_version=str(data.get("chunker_version", "")),
            extract_version=str(data.get("extract_version", "")),
            error=str(data.get("error", "")),
        )


@dataclass
class TickerState:
    """State record for a ticker workspace.

    Attributes:
        ticker: Directory-safe symbol this state belongs to.
        last_updated: When any part of this record last changed.
        documents: Per-filing index state, keyed by `doc_id`.
        catalog_checked_at: When the catalogue was last fetched successfully.
            Catalogue metadata in a record that is otherwise about the vector
            index, because `state.json` is already the one file a run reads per
            ticker and a second file would double the IO of a steady-state
            fifty-ticker pass for no behavioural gain.
        catalog_selection: The per-class document counts that fetch asked for.
            Held beside the timestamp because a run that now wants two annual
            reports must re-fetch even inside the window that a run wanting one
            could have skipped.
    """

    ticker: str
    last_updated: str
    documents: dict[str, DocumentIndexState] = field(default_factory=dict)
    catalog_checked_at: str = ""
    catalog_selection: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "last_updated": self.last_updated,
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "catalog_checked_at": self.catalog_checked_at,
            "catalog_selection": dict(self.catalog_selection),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickerState:
        ticker = str(data.get("ticker", ""))
        last_updated = str(data.get("last_updated", utc_now()))
        raw_docs = data.get("documents") or {}
        docs = {
            doc_id: DocumentIndexState.from_dict(doc_state)
            for doc_id, doc_state in raw_docs.items()
            if isinstance(doc_state, dict)
        }
        raw_selection = data.get("catalog_selection") or {}
        selection = {
            str(k): int(v)
            for k, v in raw_selection.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return cls(
            ticker=ticker,
            last_updated=last_updated,
            documents=docs,
            catalog_checked_at=str(data.get("catalog_checked_at", "")),
            catalog_selection=selection,
        )


@dataclass
class IndexResult:
    """Result of an indexing operation for a document."""

    ticker: str
    doc_id: str
    status: str  # "INDEXED", "SKIPPED", "FAILED"
    chunks_indexed: int = 0
    fingerprint: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0


class QdrantVectorIndexer:
    """Vector database indexer and state engine backed by Qdrant and SentenceTransformers."""

    def __init__(self, config: IndexerConfig | None = None) -> None:
        self.config = config or IndexerConfig.from_env()
        self._client: QdrantClient | None = None
        self._model: SentenceTransformer | None = None
        self._vector_size: int | None = None

    @property
    def client(self) -> QdrantClient:
        """Lazily initializes and returns the Qdrant client connection."""
        if self._client is None:
            if self.config.qdrant_url:
                logger.info(
                    "Connecting to Qdrant cluster at %s", self.config.qdrant_url
                )
                self._client = self._connect(self.config.prefer_grpc)
            else:
                logger.error(
                    "No QDRANT_API_URL / QDRANT_URL set. Falling back to an "
                    "in-memory Qdrant: every vector is lost when this process exits."
                )
                self._client = QdrantClient(location=":memory:")
        return self._client

    def _connect(self, prefer_grpc: bool) -> QdrantClient:
        """Opens a client, preferring gRPC but proving it before returning.

        `QdrantClient` does not dial on construction, so a cluster with no gRPC
        port would raise on the first real call -- in the middle of a ticker,
        after its extraction had already been paid for. One cheap round trip
        here turns that into a logged fallback at connect time instead.

        Args:
            prefer_grpc: Whether to try the binary transport first.

        Returns:
            A client that has answered at least one request.
        """
        if prefer_grpc:
            try:
                client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                    timeout=60.0,
                    prefer_grpc=True,
                )
                client.get_collections()
                logger.info("Qdrant transport: gRPC.")
                return client
            except Exception as exc:  # noqa: BLE001 - transport is not correctness
                logger.warning(
                    "Qdrant gRPC unavailable (%s); falling back to REST.", exc
                )
        client = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            timeout=60.0,
        )
        logger.info("Qdrant transport: REST.")
        return client

    @property
    def is_persistent(self) -> bool:
        """Whether vectors outlive this process (a server URL is configured)."""
        return bool(self.config.qdrant_url)

    @property
    def model(self) -> Any:
        """Lazily loads and returns the SentenceTransformer embedding model."""
        if self._model is None:
            logger.info("Loading embedding model %s...", self.config.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                trust_remote_code=True,
            )
            if self.config.fp16:
                # Cosine similarity over L2-normalised vectors is insensitive
                # to the last few mantissa bits, so half precision costs
                # nothing in retrieval quality and roughly halves both the
                # resident weights and the time per batch on a CUDA card.
                try:
                    self._model = self._model.half()
                    logger.info("Embedding model running in float16.")
                except Exception as exc:  # noqa: BLE001 - precision is optional
                    logger.warning("Could not switch the model to float16: %s", exc)
            logger.info(
                "Embedding device: %s (batch size %d).",
                getattr(self._model, "device", self.config.device or "auto"),
                self.config.batch_size,
            )
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    self._vector_size = self._model.get_embedding_dimension()
                else:
                    self._vector_size = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "Embedding model %s successfully loaded (vector dimension: %d).",
                    self.config.model_name,
                    self._vector_size,
                )
            except Exception:
                self._vector_size = self.config.vector_size
        return self._model

    @property
    def vector_size(self) -> int:
        """Returns the active embedding vector dimension."""
        if self._vector_size is None:
            try:
                if hasattr(self.model, "get_embedding_dimension"):
                    self._vector_size = self.model.get_embedding_dimension()
                else:
                    self._vector_size = self.model.get_sentence_embedding_dimension()
            except Exception:
                self._vector_size = self.config.vector_size
        return self._vector_size or self.config.vector_size

    def init_collection(
        self, recreate: bool = False, ticker: str | None = None
    ) -> bool:
        """Ensures a Qdrant vector collection and its payload indexes exist.

        Args:
            recreate: If True, deletes existing collection and creates fresh.
            ticker: Owner of the collection in per-ticker mode; None means the
                shared collection.

        Returns:
            bool: True if collection is ready.
        """
        collection_name = self.config.collection_for(ticker)
        target_size = self.vector_size
        exists = self.client.collection_exists(collection_name=collection_name)

        if exists and not recreate:
            try:
                coll_info = self.client.get_collection(collection_name=collection_name)
                vectors_cfg = coll_info.config.params.vectors
                existing_dim = None
                if isinstance(vectors_cfg, models.VectorParams):
                    existing_dim = vectors_cfg.size
                elif isinstance(vectors_cfg, dict) and "size" in vectors_cfg:
                    existing_dim = vectors_cfg["size"]
                if existing_dim and existing_dim != target_size:
                    logger.warning(
                        "Collection '%s' dimension mismatch (existing: %d, current model: %d). Recreating...",
                        collection_name,
                        existing_dim,
                        target_size,
                    )
                    recreate = True
            except Exception as exc:
                logger.debug("Could not verify existing collection parameters: %s", exc)

        if exists and recreate:
            logger.warning("Recreating collection '%s'...", collection_name)
            self.client.delete_collection(collection_name=collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' (vector_size=%d, distance=%s)...",
                collection_name,
                target_size,
                self.config.distance,
            )
            dist = getattr(
                models.Distance, self.config.distance.upper(), models.Distance.COSINE
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=target_size,
                    distance=dist,
                ),
            )

        self._create_payload_indexes(collection_name)
        return True

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Provisions schema indexes on key payload fields for fast filtered search."""
        index_fields = [
            ("ticker", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("doc_type", models.PayloadSchemaType.KEYWORD),
            ("element_type", models.PayloadSchemaType.KEYWORD),
            ("page_start", models.PayloadSchemaType.INTEGER),
            ("page_end", models.PayloadSchemaType.INTEGER),
        ]
        for field_name, schema_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as exc:
                # Qdrant client may throw if index already exists
                logger.debug("Payload index creation for '%s': %s", field_name, exc)

    def get_state_file_path(self, ticker: str) -> Path:
        """Returns the absolute path to a ticker's state.json file."""
        folder = self.config.output_dir / safe_ticker(ticker)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / self.config.state_filename

    def load_ticker_state(self, ticker: str) -> TickerState:
        """Loads state.json for the specified ticker, returning empty state if absent."""
        state_path = self.get_state_file_path(ticker)
        if not state_path.exists():
            return TickerState(ticker=safe_ticker(ticker), last_updated=utc_now())

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return TickerState.from_dict(data)
        except Exception as exc:
            logger.warning("Could not read state file at %s: %s", state_path, exc)
            return TickerState(ticker=safe_ticker(ticker), last_updated=utc_now())

    def save_ticker_state(self, state: TickerState) -> None:
        """Atomically persists the ticker state to state.json."""
        state_path = self.get_state_file_path(state.ticker)
        state.last_updated = utc_now()
        temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
        try:
            temp_path.write_text(
                json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(state_path)
        except Exception as exc:
            logger.error("Failed to save state file at %s: %s", state_path, exc)
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            raise

    def should_index_document(
        self,
        ticker: str,
        doc_id: str,
        fingerprint: str,
        state: TickerState | None = None,
        force: bool = False,
    ) -> bool:
        """Determines whether a document needs vector indexing or can be skipped.

        Args:
            ticker: Stock ticker.
            doc_id: Unique document identifier.
            fingerprint: SHA-256 content fingerprint of the chunk set.
            state: Pre-loaded TickerState (optional).
            force: If True, always re-indexes regardless of state.

        Returns:
            bool: True if indexing should run, False to skip.
        """
        if force:
            return True

        curr_state = state or self.load_ticker_state(ticker)
        doc_state = curr_state.documents.get(doc_id)
        if not doc_state:
            return True

        if doc_state.status != "INDEXED":
            return True

        if doc_state.fingerprint != fingerprint:
            return True

        # Model or collection changes invalidate the vector index
        model_match = (
            doc_state.embedding_model == self.config.model_name
            or doc_state.embedding_model == self.config.model_alias
        )
        if not model_match:
            return True

        # Everything recorded still matches: index only if the vectors live
        # somewhere other than the collection this config would write to.
        return doc_state.qdrant_collection != self.config.collection_for(ticker)

    def format_chunk_embed_text(
        self,
        chunk: Chunk | dict[str, Any],
        ticker: str,
        doc_id: str,  # noqa: ARG002 - kept for call-site symmetry with build_chunk_payload
        doc_type: str = "",
        label: str = "",
    ) -> str:
        """Constructs rich contextual text representation for dense embedding."""
        if isinstance(chunk, Chunk):
            if chunk.embed_text and chunk.embed_text.strip():
                return chunk.embed_text.strip()
            element_type = chunk.element_type
            page = chunk.page_start
            trail = chunk.heading_trail
            text = chunk.text
            fig = chunk.figure
            fig_kind = (fig.kind if fig else chunk.metadata.get("kind", "")) or "chart"
            caption = (fig.caption if fig else chunk.metadata.get("caption", "")) or ""
        else:
            embed_text = str(chunk.get("embed_text") or "").strip()
            if embed_text:
                return embed_text
            element_type = str(chunk.get("element_type", ELEMENT_TEXT))
            page = int(chunk.get("page_start", 0))
            path_list = chunk.get("path") or []
            trail = " > ".join(str(p) for p in path_list)
            text = str(chunk.get("text", ""))
            meta = chunk.get("metadata") or {}
            fig_kind = str(meta.get("kind") or "")
            caption = str(meta.get("caption") or "")

        header = f"[{ticker} | {doc_type} | {label} | Page {page}"

        if element_type == ELEMENT_TABLE:
            table_tag = f"{header} | Table]\n"
            if trail:
                table_tag += f"Section: {trail}\n\n"
            return f"{table_tag}{text}".strip()

        if element_type == ELEMENT_FIGURE:
            kind_lbl = fig_kind.replace("_", " ").title() if fig_kind else "Figure"
            fig_tag = f"{header} | Figure: {fig_kind}]\n"
            if trail:
                fig_tag += f"Section: {trail}\n\n"
            cap_desc = caption or "Chart"
            return f"{fig_tag}Figure: {kind_lbl}. Caption: {cap_desc}".strip()

        # Text Element
        txt_tag = f"{header}]\n"
        if trail:
            txt_tag += f"Section: {trail}\n\n"
        return f"{txt_tag}{text}".strip()

    def build_chunk_payload(
        self,
        chunk: Chunk | dict[str, Any],
        ticker: str,
        doc_id: str,
        doc_type: str = "",
        label: str = "",
        prev_chunk_id: str = "",
        next_chunk_id: str = "",
    ) -> dict[str, Any]:
        """Constructs a comprehensive, JSON-serializable Qdrant payload dictionary.

        The embedded text is deliberately not stored: it is the chunk's
        ``content`` plus a header, nothing reads it back, and at fifty tickers
        it would double the payload footprint.
        """
        if isinstance(chunk, Chunk):
            cid = chunk.chunk_id
            etype = chunk.element_type
            p_start = chunk.page_start
            p_end = chunk.page_end
            section = chunk.section
            path = list(chunk.path)
            trail = chunk.heading_trail
            content = chunk.text
            fig_path = chunk.figure_path or (chunk.figure.path if chunk.figure else "")
            fig_kind = (
                chunk.figure.kind if chunk.figure else chunk.metadata.get("kind", "")
            ) or ""
            fig_caption = (
                chunk.figure.caption
                if chunk.figure
                else chunk.metadata.get("caption", "")
            ) or ""
            table_headers: list[str] = []
            if chunk.table and chunk.table.rows and chunk.table.header_rows > 0:
                table_headers = [str(col) for col in chunk.table.rows[0]]
            meta_dict = dict(chunk.metadata or {})
        else:
            cid = str(chunk.get("chunk_id", ""))
            etype = str(chunk.get("element_type", ELEMENT_TEXT))
            p_start = int(chunk.get("page_start", 0))
            p_end = int(chunk.get("page_end", p_start))
            section = str(chunk.get("section", ""))
            path = [str(p) for p in (chunk.get("path") or [])]
            trail = " > ".join(path)
            content = str(chunk.get("text", ""))
            meta_dict = dict(chunk.get("metadata") or {})
            fig_path = str(
                chunk.get("figure_path") or meta_dict.get("figure_path") or ""
            )
            fig_kind = str(meta_dict.get("kind") or "")
            fig_caption = str(meta_dict.get("caption") or "")
            table_headers = []
            tbl_info = chunk.get("table")
            if isinstance(tbl_info, dict) and tbl_info.get("rows"):
                hrows = int(tbl_info.get("header_rows", 1))
                if hrows > 0 and len(tbl_info["rows"]) > 0:
                    table_headers = [str(c) for c in tbl_info["rows"][0]]

        clean_meta = {}
        for k, v in meta_dict.items():
            if isinstance(v, (str, int, float, bool, list, dict)) and v is not None:
                clean_meta[k] = v

        return {
            "ticker": ticker,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "label": label,
            "chunk_id": cid,
            "element_type": etype,
            "page_start": p_start,
            "page_end": p_end,
            "section": section,
            "section_breadcrumb": trail,
            "path": path,
            "content": content,
            "table_headers": table_headers,
            "image_path": fig_path,
            "caption": fig_caption,
            "figure_kind": fig_kind,
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": next_chunk_id,
            "char_count": len(content),
            "metadata": clean_meta,
        }

    def embed_matrix(
        self,
        texts: Sequence[str],
        task: str = "retrieval.passage",
        is_query: bool = False,
    ) -> Any:
        """Embeds texts and returns the raw ``(n, dim)`` float matrix.

        The form `index_chunk_set` wants: `upload_vectors` hands the matrix
        straight to Qdrant, so nothing is converted to Python floats on the way.
        `embed_texts` wraps this for callers that do want lists.

        Args:
            texts: Strings to embed.
            task: Task adapter name, for models that take one.
            is_query: Apply the model's query prompt rather than its passage one.

        Returns:
            The embedding matrix as the sentence-transformer produced it.
        """
        return self._encode(texts, task=task, is_query=is_query)

    def embed_texts(
        self,
        texts: Sequence[str],
        task: str = "retrieval.passage",
        is_query: bool = False,
    ) -> list[list[float]]:
        """Generates L2-normalized dense embeddings for a batch of strings.

        Handles model-specific query prefixes and asymmetric retrieval adapters
        (e.g., Snowflake-Arctic prompt tuning and Jina LoRA task adapters).
        """
        vectors = self._encode(texts, task=task, is_query=is_query)
        return [] if len(vectors) == 0 else vectors.tolist()

    def _encode(
        self,
        texts: Sequence[str],
        task: str = "retrieval.passage",
        is_query: bool = False,
    ) -> Any:
        """Applies the model's prompt convention and encodes, returning a matrix."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.vector_size), dtype=np.float32)

        model_name_lower = self.config.model_name.lower()
        formatted_texts = list(texts)

        # Handle Snowflake Arctic query prompting
        if is_query and "snowflake" in model_name_lower:
            prefix = "Represent this sentence for searching relevant passages: "
            formatted_texts = [
                t if t.startswith(prefix) else f"{prefix}{t}" for t in formatted_texts
            ]
        elif is_query and "nomic" in model_name_lower:
            formatted_texts = [
                t if t.startswith("search_query: ") else f"search_query: {t}"
                for t in formatted_texts
            ]
        elif not is_query and "nomic" in model_name_lower:
            formatted_texts = [
                t if t.startswith("search_document: ") else f"search_document: {t}"
                for t in formatted_texts
            ]

        encode_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
        }
        if "jina" in model_name_lower:
            encode_kwargs["task"] = task

        try:
            embeddings = self.model.encode(formatted_texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("task", None)
            embeddings = self.model.encode(formatted_texts, **encode_kwargs)

        return [vec.tolist() for vec in embeddings]

    def upsert_points(
        self,
        points: Sequence[models.PointStruct],
        collection_name: str | None = None,
    ) -> int:
        """Upserts PointStructs into Qdrant in batches, waiting for each ack."""
        if not points:
            return 0

        target = collection_name or self.config.collection_name
        total = len(points)
        batch_size = self.config.upsert_batch_size
        upserted_count = 0

        # Only the final batch waits. Qdrant applies a shard's operations in
        # the order it received them, so an acknowledged last batch implies the
        # ones queued before it -- which preserves the invariant that matters:
        # `state.json` records INDEXED only once the server holds the vectors.
        # Waiting on every batch bought nothing but a round trip per 128 points,
        # which over a 3,000-chunk filing is pure latency.
        for i in range(0, total, batch_size):
            batch = list(points[i : i + batch_size])
            self.client.upsert(
                collection_name=target,
                points=batch,
                wait=(i + batch_size >= total),
            )
            upserted_count += len(batch)

        return upserted_count

    def upload_vectors(
        self,
        vectors: Any,
        ids: Sequence[str],
        payloads: Sequence[dict[str, Any]],
        collection_name: str | None = None,
    ) -> int:
        """Uploads an embedding matrix without materialising it in Python.

        The bulk path, and the reason it exists: building `PointStruct`s means
        turning an ``(n, 768)`` float32 matrix into ``n * 768`` Python floats --
        for one annual report some 2.4 million interpreter objects, allocated in
        a Python loop, on a machine already short of memory. `upload_collection`
        takes the matrix as it stands and batches it itself.

        Args:
            vectors: The embedding matrix, one row per id.
            ids: Deterministic point ids, aligned with the matrix rows.
            payloads: Payload dicts, aligned with the matrix rows.
            collection_name: Target collection; defaults to the configured one.

        Returns:
            The number of rows uploaded.
        """
        if len(ids) == 0:
            return 0

        target = collection_name or self.config.collection_name
        self.client.upload_collection(
            collection_name=target,
            vectors=vectors,
            payload=list(payloads),
            ids=list(ids),
            batch_size=self.config.upsert_batch_size,
            # One worker: parallel>1 forks, and each child would re-import torch
            # and open its own client for what is a network-bound upload.
            parallel=1,
            wait=True,
        )
        return len(ids)

    def count_points(self, ticker: str | None = None) -> int:
        """Returns the number of points stored for a ticker, as Qdrant reports it.

        In per-ticker mode this is the whole collection; in shared mode it is a
        filtered count. Returns 0 for a collection that does not exist yet.
        """
        collection_name = self.config.collection_for(ticker)
        if not self.client.collection_exists(collection_name=collection_name):
            return 0
        count_filter = None
        if ticker and not self.config.collection_per_ticker:
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchValue(value=safe_ticker(ticker)),
                    )
                ]
            )
        return int(
            self.client.count(
                collection_name=collection_name,
                count_filter=count_filter,
                exact=True,
            ).count
        )

    def write_local_vector_cache(self, ticker: str) -> tuple[Path, int]:
        """Mirrors a ticker's vectors from Qdrant into ``output/<TICKER>/``.

        Writes ``vectors.npz`` (``vectors`` float32 matrix, ``chunk_ids``) and
        ``payloads.json`` (one projected payload per row), the two files the
        retriever's Tier-1 matrix search reads. Each stock's embeddings thereby
        exist as one self-contained, portable artifact on disk, independent of
        the Qdrant server they were also written to.

        Args:
            ticker: Whose vectors to mirror.

        Returns:
            The vectors file path and the number of rows written. Zero rows
            removes any stale cache rather than writing an empty one.
        """
        import numpy as np

        sym = safe_ticker(ticker)
        collection_name = self.config.collection_for(sym)
        folder = self.config.output_dir / sym
        folder.mkdir(parents=True, exist_ok=True)
        vec_path = folder / LOCAL_VECTORS_FILENAME
        payload_path = folder / LOCAL_PAYLOADS_FILENAME

        scroll_filter = None
        if not self.config.collection_per_ticker:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="ticker", match=models.MatchValue(value=sym)
                    )
                ]
            )

        vectors: list[list[float]] = []
        chunk_ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        if self.client.collection_exists(collection_name=collection_name):
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=512,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in points:
                    vector = point.vector
                    if isinstance(vector, dict):  # named vectors
                        vector = next(iter(vector.values()), None)
                    if vector is None:
                        continue
                    payload = point.payload or {}
                    vectors.append([float(x) for x in vector])
                    chunk_ids.append(str(payload.get("chunk_id") or point.id))
                    payloads.append(
                        {
                            k: payload.get(k)
                            for k in LOCAL_PAYLOAD_FIELDS
                            if k in payload
                        }
                    )
                if offset is None:
                    break

        if not vectors:
            for stale in (vec_path, payload_path):
                with contextlib.suppress(OSError):
                    stale.unlink()
            return vec_path, 0

        # Deterministic row order, so two mirrors of the same store compare equal.
        order = sorted(
            range(len(chunk_ids)),
            key=lambda i: (payloads[i].get("doc_id", ""), chunk_ids[i]),
        )
        matrix = np.asarray([vectors[i] for i in order], dtype=np.float32)
        ids = np.asarray([chunk_ids[i] for i in order])
        rows = [payloads[i] for i in order]

        tmp_vec = vec_path.with_suffix(".npz.tmp")
        tmp_payload = payload_path.with_suffix(".json.tmp")
        try:
            with tmp_vec.open("wb") as handle:
                # Uncompressed deliberately. zlib over a float32 embedding
                # matrix is single-threaded CPU for almost no saving -- the
                # values are high-entropy -- and this file is written once per
                # ticker and then memory-mapped by the retriever's fast path.
                np.savez(handle, vectors=matrix, chunk_ids=ids)
            tmp_payload.write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
            tmp_vec.replace(vec_path)
            tmp_payload.replace(payload_path)
        except Exception:
            for tmp in (tmp_vec, tmp_payload):
                with contextlib.suppress(OSError):
                    tmp.unlink()
            raise

        logger.info(
            "[%s] Mirrored %d vectors from '%s' to %s",
            sym,
            len(rows),
            collection_name,
            vec_path,
        )
        return vec_path, len(rows)

    def index_chunk_set(
        self,
        chunk_set: ChunkSet,
        force: bool = False,
        state: TickerState | None = None,
    ) -> IndexResult:
        """Indexes a ChunkSet into Qdrant with state management and incremental caching."""
        start_time = time.perf_counter()
        ticker = safe_ticker(chunk_set.ticker)
        doc_id = chunk_set.doc_id
        chunks = chunk_set.chunks

        if not chunks:
            return IndexResult(
                ticker=ticker,
                doc_id=doc_id,
                status="SKIPPED",
                chunks_indexed=0,
                fingerprint=chunk_set.fingerprint,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Compute or verify fingerprint
        fingerprint = chunk_set.fingerprint
        if not fingerprint:
            fingerprint = sha256_text(
                "|".join(f"{c.chunk_id}:{sha256_text(c.text)}" for c in chunks)
            )

        curr_state = state or self.load_ticker_state(ticker)

        if not self.should_index_document(
            ticker, doc_id, fingerprint, state=curr_state, force=force
        ):
            logger.info(
                "[%s] %s is already up-to-date in Qdrant (%s). Skipping.",
                ticker,
                doc_id,
                self.config.collection_name,
            )
            return IndexResult(
                ticker=ticker,
                doc_id=doc_id,
                status="SKIPPED",
                chunks_indexed=0,
                fingerprint=fingerprint,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        collection_name = self.config.collection_for(ticker)
        logger.info(
            "[%s] Indexing %s (%d chunks) into Qdrant collection '%s'...",
            ticker,
            doc_id,
            len(chunks),
            collection_name,
        )

        try:
            self.init_collection(ticker=ticker)

            embed_texts: list[str] = [
                self.format_chunk_embed_text(
                    chunk=c,
                    ticker=ticker,
                    doc_id=doc_id,
                    doc_type=chunk_set.doc_type,
                    label=chunk_set.label,
                )
                for c in chunks
            ]

            # The matrix is handed to Qdrant as it stands. Building
            # `PointStruct`s here would first turn it into one Python float per
            # dimension per chunk -- millions of objects for a single filing.
            matrix = self.embed_matrix(embed_texts, task="retrieval.passage")

            point_ids: list[str] = []
            payloads: list[dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                prev_id = chunks[i - 1].chunk_id if i > 0 else ""
                next_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else ""
                point_ids.append(deterministic_uuid(ticker, doc_id, chunk.chunk_id))
                payloads.append(
                    self.build_chunk_payload(
                        chunk=chunk,
                        ticker=ticker,
                        doc_id=doc_id,
                        doc_type=chunk_set.doc_type,
                        label=chunk_set.label,
                        prev_chunk_id=prev_id,
                        next_chunk_id=next_id,
                    )
                )

            count = self.upload_vectors(
                matrix, point_ids, payloads, collection_name=collection_name
            )

            # Update ticker state
            params = chunk_set.params or {}
            curr_state.documents[doc_id] = DocumentIndexState(
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                embedding_model=self.config.model_alias,
                qdrant_collection=collection_name,
                indexed_at=utc_now(),
                status="INDEXED",
                chunker_version=str(params.get("version", "")),
                extract_version=str(params.get("extractor", "")),
            )
            self.save_ticker_state(curr_state)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "[%s] Successfully indexed %d chunks for %s in %.2fs.",
                ticker,
                count,
                doc_id,
                elapsed,
            )
            return IndexResult(
                ticker=ticker,
                doc_id=doc_id,
                status="INDEXED",
                chunks_indexed=count,
                fingerprint=fingerprint,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "[%s] Failed to index %s: %s", ticker, doc_id, exc, exc_info=True
            )
            # Record the failure rather than leaving no trace. Layer 1 already
            # re-processes anything not recorded as INDEXED, so this changes no
            # decision -- but without it a document that failed to embed reads
            # back as never seen, and the next run reports a retry as "new".
            params = chunk_set.params or {}
            curr_state.documents[doc_id] = DocumentIndexState(
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                embedding_model=self.config.model_alias,
                qdrant_collection=self.config.collection_for(ticker),
                indexed_at=utc_now(),
                status="FAILED",
                chunker_version=str(params.get("version", "")),
                extract_version=str(params.get("extractor", "")),
                error=str(exc)[:500],
            )
            with contextlib.suppress(Exception):
                # A state write that fails must not mask the indexing error.
                self.save_ticker_state(curr_state)
            return IndexResult(
                ticker=ticker,
                doc_id=doc_id,
                status="FAILED",
                chunks_indexed=0,
                fingerprint=fingerprint,
                error=str(exc),
                elapsed_seconds=elapsed,
            )

    def index_chunk_file(
        self, file_path: Path | str, force: bool = False
    ) -> IndexResult:
        """Loads a chunk JSON file from disk and indexes it into Qdrant."""
        path = Path(file_path)
        if not path.exists():
            return IndexResult(
                ticker="",
                doc_id=path.stem,
                status="FAILED",
                error=f"File not found: {path}",
            )

        chunk_set = read_chunk_cache(path)
        if chunk_set is None:
            # Fallback raw json load
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                chunk_set = ChunkSet.from_dict(data)
            except Exception as exc:
                return IndexResult(
                    ticker="",
                    doc_id=path.stem,
                    status="FAILED",
                    error=f"Could not parse chunk JSON at {path}: {exc}",
                )

        return self.index_chunk_set(chunk_set=chunk_set, force=force)

    def index_ticker_documents(
        self, ticker: str, force: bool = False
    ) -> list[IndexResult]:
        """Indexes all chunk files present in output/<TICKER>/chunks/ into Qdrant."""
        sym = safe_ticker(ticker)
        chunks_dir = self.config.output_dir / sym / "chunks"

        if not chunks_dir.exists():
            logger.warning(
                "No chunks directory found for ticker %s at %s", sym, chunks_dir
            )
            return []

        chunk_files = sorted(chunks_dir.glob("*.json"))
        if not chunk_files:
            logger.info("No chunk JSON files found in %s", chunks_dir)
            return []

        results: list[IndexResult] = []

        logger.info(
            "[%s] Starting vector indexing for %d chunk files...",
            sym,
            len(chunk_files),
        )

        for chunk_file in chunk_files:
            res = self.index_chunk_file(chunk_file, force=force)
            results.append(res)

        return results

    def index_all_tickers(self, force: bool = False) -> dict[str, list[IndexResult]]:
        """Scans output directory and indexes filings across all available stocks."""
        out: dict[str, list[IndexResult]] = {}
        if not self.config.output_dir.exists():
            return out

        # Underscore-prefixed directories (``_nifty50``) hold cross-ticker
        # state, not a stock's workspace.
        ticker_dirs = [
            d
            for d in self.config.output_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]
        for tdir in sorted(ticker_dirs):
            ticker_name = tdir.name
            results = self.index_ticker_documents(ticker_name, force=force)
            out[ticker_name] = results

        return out

    def search_by_vector(
        self,
        query_vector: list[float],
        ticker: str | None = None,
        doc_type: str | None = None,
        element_type: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Searches Qdrant directly with a pre-computed embedding vector."""
        collection_name = self.config.collection_for(ticker)
        must_filters: list[models.FieldCondition] = []
        # In per-ticker mode the collection itself is the ticker filter.
        if ticker and not self.config.collection_per_ticker:
            must_filters.append(
                models.FieldCondition(
                    key="ticker",
                    match=models.MatchValue(value=safe_ticker(ticker)),
                )
            )
        if doc_type:
            must_filters.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type),
                )
            )
        if element_type:
            must_filters.append(
                models.FieldCondition(
                    key="element_type",
                    match=models.MatchValue(value=element_type),
                )
            )

        query_filter = models.Filter(must=must_filters) if must_filters else None

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            search_results = response.points
        elif hasattr(self.client, "search"):
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:
            response = self.client.search_points(
                collection_name=collection_name,
                vector=query_vector,
                filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            search_results = response

        formatted: list[dict[str, Any]] = []
        for hit in search_results:
            formatted.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {},
                }
            )
        return formatted

    def search_batch_by_vectors(
        self,
        requests: list[dict[str, Any]],
        with_payload: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Executes multiple vector search queries in ONE single HTTP request payload.

        Every request in a batch must name the same ticker (or none): a batch
        is one HTTP call against one collection.
        """
        if not requests:
            return []

        tickers = {safe_ticker(r["ticker"]) for r in requests if r.get("ticker")}
        if len(tickers) > 1:
            raise ValueError(f"a search batch spans one ticker, got {sorted(tickers)}")
        batch_ticker = next(iter(tickers), None)
        collection_name = self.config.collection_for(batch_ticker)

        payload_selector = (
            models.PayloadSelectorInclude(include=with_payload)
            if with_payload
            else True
        )

        search_queries: list[models.Query] = []
        for req in requests:
            query_vector = req["query_vector"]
            ticker = req.get("ticker")
            doc_types = req.get("doc_types")
            element_types = req.get("element_types")
            limit = req.get("limit", 5)
            score_threshold = req.get("score_threshold")

            must_filters: list[models.FieldCondition] = []
            if ticker and not self.config.collection_per_ticker:
                must_filters.append(
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchValue(value=safe_ticker(ticker)),
                    )
                )
            if doc_types:
                must_filters.append(
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchAny(
                            any=[d.strip().lower() for d in doc_types]
                        ),
                    )
                )
            if element_types:
                must_filters.append(
                    models.FieldCondition(
                        key="element_type",
                        match=models.MatchAny(
                            any=[e.strip().lower() for e in element_types]
                        ),
                    )
                )

            query_filter = models.Filter(must=must_filters) if must_filters else None

            search_queries.append(
                models.QueryRequest(
                    query=query_vector,
                    filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=payload_selector,
                )
            )

        if hasattr(self.client, "query_batch_points"):
            responses = self.client.query_batch_points(
                collection_name=collection_name,
                requests=search_queries,
            )
            batch_results = [resp.points for resp in responses]
        elif hasattr(self.client, "search_batch"):
            search_reqs = [
                models.SearchRequest(
                    vector=req["query_vector"],
                    filter=q.filter,
                    limit=q.limit,
                    score_threshold=q.score_threshold,
                    with_payload=payload_selector,
                )
                for req, q in zip(requests, search_queries, strict=True)
            ]
            responses = self.client.search_batch(
                collection_name=collection_name,
                requests=search_reqs,
            )
            batch_results = responses
        else:
            batch_results = [
                self.search_by_vector(
                    query_vector=req["query_vector"],
                    ticker=req.get("ticker"),
                    limit=req.get("limit", 5),
                    score_threshold=req.get("score_threshold"),
                )
                for req in requests
            ]

        formatted_batch: list[list[dict[str, Any]]] = []
        for point_list in batch_results:
            formatted: list[dict[str, Any]] = []
            for hit in point_list:
                if isinstance(hit, dict):
                    formatted.append(hit)
                else:
                    formatted.append(
                        {
                            "id": hit.id,
                            "score": hit.score,
                            "payload": hit.payload or {},
                        }
                    )
            formatted_batch.append(formatted)

        return formatted_batch

    def search(
        self,
        query: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        element_type: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Searches the Qdrant collection using semantic vector similarity and payload filters."""
        query_vector = self.embed_texts([query], task="retrieval.query", is_query=True)[
            0
        ]
        return self.search_by_vector(
            query_vector=query_vector,
            ticker=ticker,
            doc_type=doc_type,
            element_type=element_type,
            limit=limit,
            score_threshold=score_threshold,
        )


def index_ticker_documents(
    ticker: str,
    force: bool = False,
    config: IndexerConfig | None = None,
) -> list[IndexResult]:
    """Convenience functional API to index all filings for a stock."""
    indexer = QdrantVectorIndexer(config=config)
    return indexer.index_ticker_documents(ticker=ticker, force=force)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GrowNXT Qdrant Vector Indexing Engine"
    )
    parser.add_argument(
        "--ticker", type=str, help="Specific stock ticker to index (e.g. WIPRO, TCS)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Index all stocks in output directory"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-indexing ignoring cached state"
    )
    parser.add_argument(
        "--recreate-collection", action="store_true", help="Recreate Qdrant collection"
    )
    parser.add_argument(
        "--query", type=str, help="Run a test search query against Qdrant"
    )

    args = parser.parse_args()

    cfg = IndexerConfig.from_env()
    indexer = QdrantVectorIndexer(config=cfg)

    if args.recreate_collection:
        logger.info("Recreating collection '%s'...", cfg.collection_name)
        indexer.init_collection(recreate=True)

    if args.query:
        logger.info("Executing test query: '%s' (ticker=%s)", args.query, args.ticker)
        res = indexer.search(args.query, ticker=args.ticker, limit=3)
        print(f"\n--- Search Results ({len(res)}) ---")
        for i, hit in enumerate(res, 1):
            p = hit["payload"]
            print(
                f"\n[{i}] Score: {hit['score']:.4f} | {p.get('ticker')} | {p.get('chunk_id')} ({p.get('element_type')})"
            )
            print(f"    Section: {p.get('section_breadcrumb')}")
            content_preview = str(p.get("content", ""))[:160].replace("\n", " ")
            print(f"    Content: {content_preview}...")
        sys.exit(0)

    if args.ticker:
        logger.info("Indexing ticker: %s", args.ticker)
        results = indexer.index_ticker_documents(args.ticker, force=args.force)
        for r in results:
            print(
                f"{r.ticker} | {r.doc_id}: {r.status} ({r.chunks_indexed} chunks in {r.elapsed_seconds:.2f}s)"
            )
    elif args.all:
        logger.info("Indexing all tickers in %s...", cfg.output_dir)
        all_results = indexer.index_all_tickers(force=args.force)
        for t, res_list in all_results.items():
            print(f"\n--- {t} ---")
            for r in res_list:
                print(
                    f"  {r.doc_id}: {r.status} ({r.chunks_indexed} chunks in {r.elapsed_seconds:.2f}s)"
                )
    else:
        parser.print_help()
