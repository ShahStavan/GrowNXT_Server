"""Embedding client and on-disk vector store.

Chunks are embedded with ``qwen3-embed`` through the GrowNXT hosted inference
API, which speaks the OpenAI embeddings protocol. Two decisions here are worth
stating, because both are invisible in the output if they go wrong:

* **The model identity is stored beside the vectors.** Vectors produced by two
  different models are not comparable, so a store that silently mixes them
  returns confident nonsense -- similarity scores stay in their usual range and
  the retrieved chunks are simply wrong. The store records the model, and the
  pipeline treats a model change as a reason to re-embed.
* **Vectors are stored as raw float32, not JSON.** At 4096 dimensions a single
  annual report runs to a few thousand chunks; as JSON text that is hundreds of
  megabytes and seconds of parsing per query, against 16 KB per vector and a
  single ``memmap``-able read as binary.

Vectors are L2-normalised on write, which makes cosine similarity a dot
product and keeps the search loop to one matrix multiply.

Google Python Style Guide Compliant.
"""

from concurrent import futures
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from ingestion.registry import sha256_file, utc_now

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared in requirements
    np = None
    NUMPY_AVAILABLE = False

EMBEDDER_VERSION: str = "embedder/v1"

DEFAULT_EMBED_MODEL: str = os.getenv("GROWNXT_EMBED_MODEL", "qwen3-embed")
DEFAULT_API_URL: str = os.getenv("GROWNXT_LLM_API_URL", "https://grownxt-llm.vercel.app")

# Batch size for one embeddings request. 64 chunks of a filing is about 22k
# tokens, which the service returns in roughly fifteen seconds; larger batches
# push a single request's latency up without improving throughput.
DEFAULT_BATCH_SIZE: int = 64

# Batches issued concurrently. The work is almost entirely waiting on the
# network, so a few threads cut the wall time of a large document proportionally.
# Kept modest deliberately: the endpoint is shared, and the aim is to stop one
# document monopolising it, not to saturate it.
DEFAULT_CONCURRENCY: int = int(os.getenv("GROWNXT_EMBED_CONCURRENCY", "4"))

# Vector element type. Half the memory of float64 with no measurable retrieval
# difference at these dimensions.
DTYPE: str = "float32"

VECTOR_SUFFIX: str = ".f32"
MANIFEST_SUFFIX: str = ".vectors.json"


class EmbeddingError(RuntimeError):
    """Raised when a batch cannot be embedded after retries."""


class EmbeddingRejected(EmbeddingError):
    """Raised when the service refuses a batch's content outright.

    Distinct from a transient failure because the response is a property of the
    input, not of the moment: the same bytes will be refused again, so retrying
    wastes four round trips and then abandons a document that is almost entirely
    embeddable. The caller bisects instead.
    """


# HTTP statuses that mean "this input, not this moment". 408 and 429 are
# excluded: a timeout and a rate limit are both worth retrying unchanged.
_PERMANENT_STATUSES = frozenset(
    code for code in range(400, 500) if code not in (408, 429)
)


@dataclass
class EmbeddingClient:
    """Client for the hosted OpenAI-compatible embeddings endpoint.

    Attributes:
        model: Embedding model identifier.
        api_url: Service root.
        api_key: Bearer token; read from the environment when not given.
        batch_size: Inputs per request.
        max_workers: Batches issued concurrently.
        timeout: Per-request timeout in seconds.
        max_retries: Additional attempts per batch.
    """

    model: str = DEFAULT_EMBED_MODEL
    api_url: str = DEFAULT_API_URL
    api_key: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE
    max_workers: int = DEFAULT_CONCURRENCY
    timeout: int = 180
    max_retries: int = 3
    _dim: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolves the API key from the environment when not supplied."""
        if not self.api_key:
            self.api_key = os.getenv("GROWNXT_LLM_API_KEY", "")

    @property
    def endpoint(self) -> str:
        """Returns the embeddings URL."""
        return self.api_url.rstrip("/") + "/v1/embeddings"

    @property
    def dim(self) -> int:
        """Returns the embedding dimension seen so far, or 0 before any call."""
        return self._dim

    def _headers(self) -> Dict[str, str]:
        """Returns request headers, including the bearer token when present."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def _request(self, inputs: Sequence[str]) -> List[List[float]]:
        """Embeds one batch, returning vectors in the order submitted.

        Args:
            inputs: Texts to embed.

        Returns:
            One vector per input.

        Raises:
            EmbeddingError: If the batch fails after every retry.
        """
        payload = {"model": self.model, "input": list(inputs)}
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, json=payload,
                                         headers=self._headers(), timeout=self.timeout)
                if response.ok:
                    body = response.json()
                    items = body.get("data") or []
                    if len(items) != len(inputs):
                        last_error = "expected %d vectors, received %d" % (
                            len(inputs), len(items))
                        raise ValueError(last_error)
                    # The protocol allows results out of order; index is
                    # authoritative, so never rely on arrival order.
                    ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
                    vectors = [list(item.get("embedding") or []) for item in ordered]
                    if not vectors or not vectors[0]:
                        raise ValueError("empty embedding returned")
                    return vectors
                last_error = "HTTP %d: %s" % (response.status_code, response.text[:200])
                if response.status_code in _PERMANENT_STATUSES:
                    # The content, not the moment. Fail immediately so the
                    # caller can isolate which input is being refused.
                    raise EmbeddingRejected(last_error)
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            logger.warning("Embedding batch failed (%s), attempt %d/%d",
                           last_error, attempt + 1, self.max_retries + 1)
            if attempt < self.max_retries:
                time.sleep(2.0 * (attempt + 1))
        raise EmbeddingError("Could not embed batch: " + (last_error or "unknown error"))

    def embed_tolerating_rejections(
        self,
        texts: Sequence[str],
        progress_every: int = 10,
    ) -> Tuple[Any, List[int], str]:
        """Embeds texts, isolating and excluding any the service refuses.

        The endpoint applies a content filter, and it applies it to the whole
        request: one passage it dislikes fails the batch of 64 around it. An
        issuer's risk slide naming a security incident is enough, and refusing
        the document over it would discard forty usable passages to avoid one.

        Rejected batches are therefore bisected until the offending inputs are
        isolated, which costs about log2(batch) extra requests and leaves every
        other passage embedded. The excluded indices are returned rather than
        filled with a placeholder vector: a zero row would sit in the store
        claiming to be a chunk's embedding while matching nothing, and the
        manifest would assert a correspondence that does not hold.

        Args:
            texts: Texts to embed, in order.
            progress_every: Log a progress line every N completed batches.

        Returns:
            A tuple of (matrix of the embedded rows, indices excluded, reason).
            The matrix has one row per surviving text, in input order.

        Raises:
            EmbeddingError: If a batch fails for a transient reason after every
                retry, which is not a property of the content.
            RuntimeError: If numpy is unavailable.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for embedding; install it first.")
        if not texts:
            return np.zeros((0, 0), dtype=DTYPE), [], ""

        vectors: List[Optional[List[float]]] = [None] * len(texts)
        excluded: List[int] = []
        reason = ""

        def resolve(offset: int, batch: Sequence[str]) -> None:
            """Embeds one batch, bisecting it if its content is refused."""
            nonlocal reason
            try:
                for position, vector in enumerate(self._request(batch)):
                    vectors[offset + position] = vector
                return
            except EmbeddingRejected as exc:
                if len(batch) == 1:
                    excluded.append(offset)
                    reason = reason or str(exc)[:200]
                    logger.warning("Chunk at index %d refused by the service; "
                                   "excluding it and continuing.", offset)
                    return
                middle = len(batch) // 2
                logger.info("Batch of %d refused; bisecting to isolate the cause.",
                            len(batch))
                resolve(offset, batch[:middle])
                resolve(offset + middle, batch[middle:])

        batches = [
            (start, texts[start:start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        workers = max(1, min(int(self.max_workers), len(batches)))
        if workers == 1:
            for start, batch in batches:
                resolve(start, batch)
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                pending = [pool.submit(resolve, start, batch) for start, batch in batches]
                for future in futures.as_completed(pending):
                    future.result()

        kept = [vector for vector in vectors if vector is not None]
        widths = {len(vector) for vector in kept}
        if len(widths) > 1:
            raise EmbeddingError(
                "Embedding width is inconsistent across batches (%s). The service "
                "is serving more than one model; the store would be unusable."
                % sorted(widths)
            )
        if widths:
            self._dim = widths.pop()
        if excluded:
            logger.warning("Excluded %d of %d chunks the service refused: %s",
                           len(excluded), len(texts), reason)
        array = np.asarray(kept, dtype=DTYPE) if kept else np.zeros((0, 0), dtype=DTYPE)
        return (normalize(array) if kept else array), sorted(excluded), reason

    def embed(self, texts: Sequence[str], progress_every: int = 10) -> Any:
        """Embeds a list of texts.

        Args:
            texts: Texts to embed, in order.
            progress_every: Log a progress line every N batches.

        Returns:
            A normalised float32 array of shape (len(texts), dim).

        Raises:
            EmbeddingError: If any batch fails after retries, or if the service
                returns vectors of inconsistent width.
            RuntimeError: If numpy is unavailable.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for embedding; install it first.")
        if not texts:
            return np.zeros((0, 0), dtype=DTYPE)

        batches = [
            texts[start:start + self.batch_size]
            for start in range(0, len(texts), self.batch_size)
        ]
        results = self._embed_batches(batches, progress_every)

        # Width is checked once, over every batch, rather than as each arrives:
        # with requests in flight concurrently there is no "previous" width to
        # compare against without locking, and a mid-run model switch is caught
        # just as well here -- before a single vector reaches disk.
        widths = {len(vector) for batch in results for vector in batch}
        if len(widths) > 1:
            raise EmbeddingError(
                "Embedding width is inconsistent across batches (%s). The service "
                "is serving more than one model; the store would be unusable."
                % sorted(widths)
            )
        width = widths.pop() if widths else 0
        if self._dim and width and width != self._dim:
            raise EmbeddingError(
                "Embedding width changed mid-run: %d then %d. The service is "
                "serving more than one model; the store would be unusable."
                % (self._dim, width)
            )
        self._dim = width

        array = np.asarray([vector for batch in results for vector in batch], dtype=DTYPE)
        return normalize(array)

    def _embed_batches(
        self,
        batches: Sequence[Sequence[str]],
        progress_every: int,
    ) -> List[List[List[float]]]:
        """Embeds every batch, concurrently when that is worthwhile.

        The endpoint takes seconds to return 64 chunks of a filing, and a long
        annual report is dozens of those batches. Almost
        all of that is waiting on the network, so the requests are issued from a
        small thread pool. Results are reassembled strictly by batch index: the
        vector file's rows are positional and are matched to chunk ids by
        position, so a reordering here would pair every chunk with another
        chunk's vector while leaving every count and checksum plausible.

        Args:
            batches: Batches of texts, in order.
            progress_every: Log a progress line every N completed batches.

        Returns:
            One list of vectors per batch, in the order the batches were given.

        Raises:
            EmbeddingError: If any batch fails after its retries.
        """
        total = len(batches)
        workers = max(1, min(int(self.max_workers), total))
        results: List[Optional[List[List[float]]]] = [None] * total
        done = 0

        def note_progress() -> None:
            """Logs completion progress."""
            if done % max(1, progress_every) == 0 or done == total:
                embedded = sum(len(batch) for batch in batches[:done])
                logger.info("Embedded %d/%d batches (~%d chunks)", done, total, embedded)

        if workers == 1:
            for index, batch in enumerate(batches):
                results[index] = self._request(batch)
                done += 1
                note_progress()
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(self._request, batch): index
                    for index, batch in enumerate(batches)
                }
                for future in futures.as_completed(pending):
                    index = pending[future]
                    # A failed batch has already exhausted its retries. Raising
                    # here abandons the document rather than storing a vector
                    # set with a hole in it that no later stage could detect.
                    results[index] = future.result()
                    done += 1
                    note_progress()

        missing = [index for index, value in enumerate(results) if value is None]
        if missing:
            raise EmbeddingError("Batches %s produced no vectors." % missing[:5])
        return [batch for batch in results if batch is not None]


def normalize(array: Any) -> Any:
    """Returns a copy of `array` with each row scaled to unit length.

    Normalising once on write means every later similarity is a dot product,
    which keeps retrieval to a single matrix multiply.
    """
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(DTYPE, copy=False)


@dataclass
class VectorStore:
    """A document's vectors on disk, with the manifest that describes them.

    Attributes:
        doc_id: Owning document.
        model: Model that produced the vectors.
        dim: Vector width.
        chunk_ids: Chunk identifier per row, in row order.
        fingerprint: Chunk-set fingerprint the vectors were built from.
        vectors: The array, when loaded.
        excluded_chunk_ids: Chunks the service refused to embed. Named
            rather than dropped silently: a passage missing from the index
            is unretrievable, and a reader comparing the chunk cache with
            the manifest needs to see that the gap was deliberate.
        exclusion_reason: What the service said about them.
    """

    doc_id: str
    model: str
    dim: int
    chunk_ids: List[str] = field(default_factory=list)
    fingerprint: str = ""
    vectors: Any = None
    excluded_chunk_ids: List[str] = field(default_factory=list)
    exclusion_reason: str = ""

    @property
    def count(self) -> int:
        """Number of vectors."""
        return len(self.chunk_ids)

    def manifest(self) -> Dict[str, Any]:
        """Returns the JSON manifest written beside the vector file."""
        return {
            "doc_id": self.doc_id,
            "embedder_version": EMBEDDER_VERSION,
            "model": self.model,
            "dim": self.dim,
            "count": self.count,
            "dtype": DTYPE,
            "normalized": True,
            "fingerprint": self.fingerprint,
            "chunk_ids": self.chunk_ids,
            "excluded_chunk_ids": self.excluded_chunk_ids,
            "exclusion_reason": self.exclusion_reason,
            "written_at": utc_now(),
        }


def vector_paths(directory: Path, doc_id: str) -> Tuple[Path, Path]:
    """Returns the (vector file, manifest file) paths for one document."""
    return directory / (doc_id + VECTOR_SUFFIX), directory / (doc_id + MANIFEST_SUFFIX)


def save_vectors(
    directory: Path,
    doc_id: str,
    model: str,
    chunk_ids: Sequence[str],
    array: Any,
    fingerprint: str,
    excluded_chunk_ids: Optional[Sequence[str]] = None,
    exclusion_reason: str = "",
) -> Dict[str, Any]:
    """Writes vectors and their manifest to disk.

    Args:
        directory: Target directory, normally ``data/<TICKER>/vectors``.
        doc_id: Owning document.
        model: Model that produced the vectors.
        chunk_ids: Chunk identifier per row, in row order.
        array: Normalised float32 array of shape (len(chunk_ids), dim).
        fingerprint: Chunk-set fingerprint the vectors were built from.
        excluded_chunk_ids: Chunks the service refused to embed.
        exclusion_reason: What the service said about them.

    Returns:
        Detail for the registry: relative paths, counts, and the file hash.

    Raises:
        ValueError: If the array's row count does not match `chunk_ids`.
    """
    if array.shape[0] != len(chunk_ids):
        raise ValueError("vector rows (%d) do not match chunk ids (%d)"
                         % (array.shape[0], len(chunk_ids)))
    directory.mkdir(parents=True, exist_ok=True)
    vector_file, manifest_file = vector_paths(directory, doc_id)

    store = VectorStore(
        doc_id=doc_id, model=model, dim=int(array.shape[1]),
        chunk_ids=list(chunk_ids), fingerprint=fingerprint,
        excluded_chunk_ids=list(excluded_chunk_ids or ()),
        exclusion_reason=exclusion_reason,
    )
    # Raw C-order float32: readable with one np.fromfile, and the manifest
    # carries the shape that the raw file cannot.
    array.astype(DTYPE, copy=False).tofile(str(vector_file))
    manifest_file.write_text(json.dumps(store.manifest()), encoding="utf-8")

    return {
        "model": model,
        "dim": store.dim,
        "count": store.count,
        "vectors_path": "vectors/" + vector_file.name,
        "manifest_path": "vectors/" + manifest_file.name,
        "vectors_sha256": sha256_file(vector_file),
        "bytes": vector_file.stat().st_size,
        "excluded_chunk_ids": store.excluded_chunk_ids,
        "exclusion_reason": store.exclusion_reason,
    }


def load_vectors(directory: Path, doc_id: str) -> Optional[VectorStore]:
    """Loads one document's vectors and manifest.

    Args:
        directory: Directory holding the vector files.
        doc_id: Document to load.

    Returns:
        The populated VectorStore, or None when either file is missing or the
        two disagree about the shape -- a mismatch means the pair was written by
        different runs, and reshaping it anyway would silently mis-associate
        every chunk with another chunk's vector.
    """
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required to load vectors; install it first.")
    vector_file, manifest_file = vector_paths(directory, doc_id)
    if not vector_file.exists() or not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Vector manifest unreadable at %s: %s", manifest_file, exc)
        return None

    dim = int(manifest.get("dim", 0))
    chunk_ids = [str(value) for value in (manifest.get("chunk_ids") or [])]
    if dim <= 0 or not chunk_ids:
        logger.warning("Vector manifest at %s is incomplete.", manifest_file)
        return None

    try:
        flat = np.fromfile(str(vector_file), dtype=DTYPE)
    except OSError as exc:
        logger.warning("Vector file unreadable at %s: %s", vector_file, exc)
        return None

    expected = dim * len(chunk_ids)
    if flat.size != expected:
        logger.error("Vector file %s holds %d values, manifest expects %d; "
                     "refusing to guess the shape.", vector_file, flat.size, expected)
        return None

    return VectorStore(
        doc_id=doc_id,
        model=str(manifest.get("model", "")),
        dim=dim,
        chunk_ids=chunk_ids,
        fingerprint=str(manifest.get("fingerprint", "")),
        vectors=flat.reshape(len(chunk_ids), dim),
        excluded_chunk_ids=[
            str(value) for value in (manifest.get("excluded_chunk_ids") or [])
        ],
        exclusion_reason=str(manifest.get("exclusion_reason", "")),
    )
