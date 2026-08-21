"""Retrieval over a ticker's embedded filings.

Selects the evidence that goes into an extraction prompt. Two things make this
more than a nearest-neighbour lookup:

* **Section filtering comes before scoring, not after.** A probe for "principal
  risks" run across a whole annual report will happily return the ESG report's
  climate paragraphs, which are the nearest available text when the risk
  section is not in scope. Restricting the candidate set to the sections a
  focus area actually lives in means an absent section produces no evidence --
  which the prompt then reports as an absence -- rather than plausible evidence
  from the wrong part of the document.
* **Results are spread across the document.** Adjacent chunks are nearly
  identical vectors, so an unconstrained top-k routinely returns eight
  neighbouring chunks of one paragraph and calls the topic covered. A cap per
  section keeps a fixed evidence budget spent on distinct passages.

The store is small enough to search exhaustively: one annual report is a few
thousand vectors, so a single matrix multiply beats the complexity of an
approximate index and cannot silently miss a neighbour.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ingestion.chunker import Chunk, read_chunk_cache
from ingestion.embedder import EmbeddingClient, load_vectors, normalize

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared in requirements
    np = None
    NUMPY_AVAILABLE = False


@dataclass
class Hit:
    """One retrieved chunk.

    Attributes:
        chunk: The chunk itself.
        score: Cosine similarity against the query.
        focus_id: Focus area whose probe retrieved it, when applicable.
    """

    chunk: Chunk
    score: float
    focus_id: str = ""


@dataclass
class DocumentIndex:
    """One document's chunks and vectors, loaded together.

    Attributes:
        doc_id: Document identifier.
        doc_type: Document class.
        label: Period label.
        model: Model that produced the vectors.
        chunks: Chunk by chunk id.
        order: Chunk ids in vector row order.
        vectors: Normalised vector matrix.
    """

    doc_id: str
    doc_type: str
    label: str
    model: str
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    vectors: Any = None

    @property
    def sections(self) -> List[str]:
        """Returns the section keys present in this document."""
        return sorted({chunk.section_id for chunk in self.chunks.values()})

    def search(
        self,
        query_vector: Any,
        top_k: int = 8,
        sections: Optional[Iterable[str]] = None,
        max_per_section: int = 0,
    ) -> List[Hit]:
        """Returns the closest chunks to a query vector.

        Args:
            query_vector: Normalised query vector.
            top_k: Maximum hits to return.
            sections: Restrict candidates to these section keys. An empty or
                absent value searches the whole document.
            max_per_section: Cap hits per section. Zero means no cap.

        Returns:
            Hits ordered by descending similarity.
        """
        if self.vectors is None or not self.order:
            return []

        wanted = set(sections or ())
        if wanted:
            rows = [
                index for index, chunk_id in enumerate(self.order)
                if self.chunks.get(chunk_id) is not None
                and self.chunks[chunk_id].section_id in wanted
            ]
            if not rows:
                return []
            candidates = self.vectors[rows]
        else:
            rows = list(range(len(self.order)))
            candidates = self.vectors

        # Both sides are unit vectors, so the dot product is the cosine.
        scores = candidates.dot(query_vector)
        ranked = np.argsort(-scores)

        hits: List[Hit] = []
        per_section: Dict[str, int] = {}
        for position in ranked:
            chunk_id = self.order[rows[int(position)]]
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            if max_per_section:
                seen = per_section.get(chunk.section_id, 0)
                if seen >= max_per_section:
                    continue
                per_section[chunk.section_id] = seen + 1
            hits.append(Hit(chunk=chunk, score=float(scores[int(position)])))
            if len(hits) >= top_k:
                break
        return hits


def load_document_index(
    chunks_dir: Path,
    vectors_dir: Path,
    doc_id: str,
) -> Optional[DocumentIndex]:
    """Loads one document's chunks and vectors.

    Args:
        chunks_dir: Directory holding chunk caches.
        vectors_dir: Directory holding vector files.
        doc_id: Document to load.

    Returns:
        The index, or None when either side is missing or the two disagree.
        A disagreement is refused rather than repaired: the vector file's rows
        are positional, so continuing with a mismatched chunk list would pair
        every chunk with another chunk's vector and return fluent nonsense.
    """
    chunk_set = read_chunk_cache(chunks_dir / (doc_id + ".json"))
    if chunk_set is None:
        return None
    store = load_vectors(vectors_dir, doc_id)
    if store is None:
        logger.info("[%s] no vectors on disk; document not searchable.", doc_id)
        return None

    by_id = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
    missing = [chunk_id for chunk_id in store.chunk_ids if chunk_id not in by_id]
    if missing:
        logger.error("[%s] %d vectors reference chunks absent from the cache "
                     "(e.g. %s); the two were written by different runs.",
                     doc_id, len(missing), missing[0])
        return None
    if store.fingerprint and chunk_set.fingerprint != store.fingerprint:
        logger.warning("[%s] vectors were built from a different chunk set; "
                       "re-run the embed stage.", doc_id)
        return None

    return DocumentIndex(
        doc_id=doc_id,
        doc_type=chunk_set.doc_type,
        label=chunk_set.label,
        model=store.model,
        chunks=by_id,
        order=list(store.chunk_ids),
        vectors=store.vectors,
    )


@dataclass
class TickerIndex:
    """Every searchable document of one ticker.

    Attributes:
        ticker: Owning ticker.
        documents: Index by document id.
    """

    ticker: str
    documents: Dict[str, DocumentIndex] = field(default_factory=dict)

    def of_type(self, doc_type: str) -> List[DocumentIndex]:
        """Returns the indexes of one document class, newest first."""
        selected = [index for index in self.documents.values() if index.doc_type == doc_type]
        return sorted(selected, key=lambda index: index.doc_id, reverse=True)

    def models(self) -> List[str]:
        """Returns the distinct embedding models across loaded documents."""
        return sorted({index.model for index in self.documents.values() if index.model})


def load_ticker_index(
    root: Path,
    doc_ids: Sequence[str],
) -> TickerIndex:
    """Loads every named document of one ticker.

    Args:
        root: The ticker directory, ``data/<TICKER>``.
        doc_ids: Documents to load.

    Returns:
        The ticker index, holding only the documents that loaded cleanly.
    """
    index = TickerIndex(ticker=root.name)
    chunks_dir = root / "chunks"
    vectors_dir = root / "vectors"
    for doc_id in doc_ids:
        document = load_document_index(chunks_dir, vectors_dir, doc_id)
        if document is not None:
            index.documents[doc_id] = document

    models = index.models()
    if len(models) > 1:
        # Comparing vectors from two models yields scores in the usual range and
        # rankings that are meaningless, so this must be loud.
        logger.error("[%s] loaded documents embedded with different models (%s). "
                     "Scores across them are not comparable; re-embed to one model.",
                     index.ticker, ", ".join(models))
    return index


def retrieve_for_probes(
    document: DocumentIndex,
    probes: Sequence[Tuple[Any, str]],
    client: EmbeddingClient,
    top_k: int = 6,
    max_per_section: int = 3,
) -> List[Hit]:
    """Retrieves evidence for a document's focus areas.

    Each probe is embedded and searched within its focus area's sections, and
    the results are merged, de-duplicated, and ordered by score. A chunk
    retrieved by two probes keeps the higher score and its first focus area, so
    that the evidence block never repeats a passage while a limited budget goes
    unspent.

    Args:
        document: Index to search.
        probes: Pairs of (focus area, query text).
        client: Embedding client, used for the query vectors.
        top_k: Hits per probe.
        max_per_section: Cap on hits from one section per probe.

    Returns:
        Merged hits, highest score first.
    """
    if not probes:
        return []
    queries = [text for _area, text in probes]
    matrix = client.embed(queries, progress_every=1000)
    if matrix.shape[0] != len(probes):
        logger.error("[%s] embedded %d probes but expected %d; skipping retrieval.",
                     document.doc_id, matrix.shape[0], len(probes))
        return []

    best: Dict[str, Hit] = {}
    for row, (area, _text) in enumerate(probes):
        available = set(document.sections)
        sections = [key for key in getattr(area, "sections", []) if key in available]
        hits = document.search(
            matrix[row], top_k=top_k,
            sections=sections or None, max_per_section=max_per_section,
        )
        focus_id = getattr(area, "focus_id", "")
        for hit in hits:
            hit.focus_id = focus_id
            current = best.get(hit.chunk.chunk_id)
            if current is None or hit.score > current.score:
                best[hit.chunk.chunk_id] = hit

    merged = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
    logger.info("[%s] retrieved %d distinct chunks across %d probes",
                document.doc_id, len(merged), len(probes))
    return merged


def embed_query(client: EmbeddingClient, text: str) -> Any:
    """Embeds a single query string and returns its normalised vector."""
    matrix = client.embed([text], progress_every=1000)
    return normalize(matrix)[0]
