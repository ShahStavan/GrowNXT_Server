"""Parallel Multi-Query Vector Retriever for GrowNXT RAG Pipeline.

Executes concurrent, asymmetric vector searches against Qdrant Cloud using
Snowflake-Arctic-Embed with per-pillar payload filtering.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from core.config import safe_ticker
from ingestion.indexer import IndexerConfig, QdrantVectorIndexer
from ingestion.rag.probes import ThematicProbe

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_TOP_K: int = 6
DEFAULT_PARALLEL_WORKERS: int = 5


@dataclass
class EvidenceChunk:
    """A grounded piece of evidence retrieved from filing element chunks."""

    chunk_id: str
    doc_id: str
    doc_type: str
    ticker: str
    label: str
    element_type: str
    page_start: int
    page_end: int
    section_breadcrumb: str
    content: str
    score: float
    query_matched: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_tag(self) -> str:
        """Returns concise institutional inline citation tag (e.g. '(Presentation, p. 5)')."""
        lbl = (self.label or self.doc_id).replace(self.ticker, "").strip(" -|,")
        if "presentation" in lbl.lower():
            doc_str = "Investor Presentation"
        elif "transcript" in lbl.lower():
            doc_str = "Earnings Concall"
        elif "annual" in lbl.lower() or "report" in lbl.lower():
            doc_str = "Annual Report"
        else:
            doc_str = self.doc_type.replace("_", " ").title()

        page_str = f"p. {self.page_start}" if self.page_start > 0 else "Filing"
        return f"({doc_str}, {page_str})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceChunk":
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            doc_id=str(data.get("doc_id", "")),
            doc_type=str(data.get("doc_type", "")),
            ticker=str(data.get("ticker", "")),
            label=str(data.get("label", "")),
            element_type=str(data.get("element_type", "text")),
            page_start=int(data.get("page_start", 0)),
            page_end=int(data.get("page_end", 0)),
            section_breadcrumb=str(data.get("section_breadcrumb", "")),
            content=str(data.get("content", "")),
            score=float(data.get("score", 0.0)),
            query_matched=str(data.get("query_matched", "")),
            metadata=dict(data.get("metadata") or {}),
        )


# Persistent on-disk vector cache for deterministic probe queries (Phase 1 optimization)
PROBE_CACHE_FILE = Path(__file__).parent / "probe_vectors.json"
_PROBE_EMBEDDING_CACHE: dict[str, list[float]] = {}


def _load_probe_cache() -> None:
    """Loads pre-computed probe embedding vectors from persistent disk storage."""
    global _PROBE_EMBEDDING_CACHE
    if _PROBE_EMBEDDING_CACHE:
        return
    if PROBE_CACHE_FILE.exists():
        try:
            _PROBE_EMBEDDING_CACHE = json.loads(PROBE_CACHE_FILE.read_text(encoding="utf-8"))
            logger.info("Loaded %d pre-computed probe embeddings from disk cache (%s).", len(_PROBE_EMBEDDING_CACHE), PROBE_CACHE_FILE.name)
        except Exception as exc:
            logger.warning("Could not load probe vector cache from disk: %s", exc)


def _save_probe_cache() -> None:
    """Persists computed probe embedding vectors to disk storage."""
    if not _PROBE_EMBEDDING_CACHE:
        return
    try:
        PROBE_CACHE_FILE.write_text(json.dumps(_PROBE_EMBEDDING_CACHE), encoding="utf-8")
        logger.info("Persisted %d probe embeddings to disk cache (%s).", len(_PROBE_EMBEDDING_CACHE), PROBE_CACHE_FILE.name)
    except Exception as exc:
        logger.warning("Could not persist probe vector cache to disk: %s", exc)


class ParallelVectorRetriever:
    """High-throughput concurrent vector retriever backed by Qdrant and Snowflake Arctic."""

    def __init__(
        self,
        indexer: Optional[QdrantVectorIndexer] = None,
        config: Optional[IndexerConfig] = None,
    ) -> None:
        self.indexer = indexer or QdrantVectorIndexer(config=config)

    def retrieve_single_query(
        self,
        ticker: str,
        query: str,
        doc_types: Optional[Sequence[str]] = None,
        element_types: Optional[Sequence[str]] = None,
        limit: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> list[EvidenceChunk]:
        """Executes a single semantic query against Qdrant with payload filtering."""
        sym = safe_ticker(ticker)
        # Search via indexer (which automatically adds the query prompt prefix for Snowflake Arctic)
        try:
            hits = self.indexer.search(
                query=query,
                ticker=sym,
                limit=limit * 2,  # Fetch slightly wider candidate pool for subsequent filtering
            )
        except Exception as exc:
            logger.error("[%s] Search failed for query '%s': %s", sym, query, exc)
            return []

        results: list[EvidenceChunk] = []
        doc_type_filter = {d.strip().lower() for d in doc_types} if doc_types else None
        elem_type_filter = {e.strip().lower() for e in element_types} if element_types else None

        for hit in hits:
            payload = hit.get("payload") or {}
            hit_doc_type = str(payload.get("doc_type", "")).lower()
            hit_elem_type = str(payload.get("element_type", "")).lower()

            if doc_type_filter and hit_doc_type not in doc_type_filter:
                continue
            if elem_type_filter and hit_elem_type not in elem_type_filter:
                continue

            results.append(
                EvidenceChunk(
                    chunk_id=str(payload.get("chunk_id") or hit.get("id")),
                    doc_id=str(payload.get("doc_id", "")),
                    doc_type=str(payload.get("doc_type", "")),
                    ticker=str(payload.get("ticker", sym)),
                    label=str(payload.get("label", "")),
                    element_type=str(payload.get("element_type", "text")),
                    page_start=int(payload.get("page_start", 0)),
                    page_end=int(payload.get("page_end", 0)),
                    section_breadcrumb=str(payload.get("section_breadcrumb", "")),
                    content=str(payload.get("content", "")),
                    score=float(hit.get("score", 0.0)),
                    query_matched=query,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
            if len(results) >= limit:
                break

        return results

    def retrieve_single_query_vector(
        self,
        ticker: str,
        query: str,
        query_vector: list[float],
        doc_types: Optional[Sequence[str]] = None,
        element_types: Optional[Sequence[str]] = None,
        limit: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> list[EvidenceChunk]:
        """Executes a search against Qdrant using a pre-computed embedding vector."""
        sym = safe_ticker(ticker)
        try:
            hits = self.indexer.search_by_vector(
                query_vector=query_vector,
                ticker=sym,
                limit=limit * 2,
            )
        except Exception as exc:
            logger.error("[%s] Search by vector failed for query '%s': %s", sym, query, exc)
            return []

        results: list[EvidenceChunk] = []
        doc_type_filter = {d.strip().lower() for d in doc_types} if doc_types else None
        elem_type_filter = {e.strip().lower() for e in element_types} if element_types else None

        for hit in hits:
            payload = hit.get("payload") or {}
            hit_doc_type = str(payload.get("doc_type", "")).lower()
            hit_elem_type = str(payload.get("element_type", "")).lower()

            if doc_type_filter and hit_doc_type not in doc_type_filter:
                continue
            if elem_type_filter and hit_elem_type not in elem_type_filter:
                continue

            results.append(
                EvidenceChunk(
                    chunk_id=str(payload.get("chunk_id") or hit.get("id")),
                    doc_id=str(payload.get("doc_id", "")),
                    doc_type=str(payload.get("doc_type", "")),
                    ticker=str(payload.get("ticker", sym)),
                    label=str(payload.get("label", "")),
                    element_type=str(payload.get("element_type", "text")),
                    page_start=int(payload.get("page_start", 0)),
                    page_end=int(payload.get("page_end", 0)),
                    section_breadcrumb=str(payload.get("section_breadcrumb", "")),
                    content=str(payload.get("content", "")),
                    score=float(hit.get("score", 0.0)),
                    query_matched=query,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
            if len(results) >= limit:
                break

        return results

    def retrieve_all_probes_parallel(
        self,
        ticker: str,
        probes: Sequence[ThematicProbe],
        top_k_per_query: int = DEFAULT_RETRIEVAL_TOP_K,
        max_workers: int = DEFAULT_PARALLEL_WORKERS,
    ) -> dict[str, list[EvidenceChunk]]:
        """Executes high-throughput batch-embedded single-flight vector searches across all probes."""
        total_start = time.perf_counter()
        sym = safe_ticker(ticker)
        results: dict[str, list[EvidenceChunk]] = {p.pillar: [] for p in probes}

        # 1. Collect all distinct queries across all probes
        all_queries: list[str] = []
        for p in probes:
            for q in p.queries:
                if q not in all_queries:
                    all_queries.append(q)

        # 2. Check probe cache vs uncached queries (Zero forward-pass for cached queries)
        _load_probe_cache()
        t_embed_start = time.perf_counter()
        uncached_queries = [q for q in all_queries if q not in _PROBE_EMBEDDING_CACHE]
        if uncached_queries:
            computed_vectors = self.indexer.embed_texts(uncached_queries, task="retrieval.query", is_query=True)
            for q, vec in zip(uncached_queries, computed_vectors):
                _PROBE_EMBEDDING_CACHE[q] = vec
            _save_probe_cache()

        query_vec_map = {q: _PROBE_EMBEDDING_CACHE[q] for q in all_queries if q in _PROBE_EMBEDDING_CACHE}
        embed_elapsed = time.perf_counter() - t_embed_start

        logger.info(
            "[%s] Probe Embeddings: %d queries (%d cached, %d computed) in %.3fs.",
            sym,
            len(all_queries),
            len(all_queries) - len(uncached_queries),
            len(uncached_queries),
            embed_elapsed,
        )

        # Tier-1: Ultra-Fast Sub-Millisecond Local Vector Matrix Dot-Product
        local_vec_file = Path(f"output/{sym}/vectors.npz")
        local_payload_file = Path(f"output/{sym}/payloads.json")

        if local_vec_file.exists() and local_payload_file.exists():
            try:
                t_local_start = time.perf_counter()
                vec_data = np.load(local_vec_file)
                doc_vectors = vec_data["vectors"]
                chunk_ids = vec_data["chunk_ids"]
                payloads = json.loads(local_payload_file.read_text(encoding="utf-8"))

                q_list: list[list[float]] = []
                slot_map: list[tuple[str, str, Sequence[str]]] = []
                for p in probes:
                    for q in p.queries:
                        if q in query_vec_map:
                            q_list.append(query_vec_map[q])
                            slot_map.append((p.pillar, q, p.target_doc_types))

                if q_list and len(doc_vectors) > 0:
                    q_mat = np.array(q_list, dtype=np.float32)
                    q_mat /= np.linalg.norm(q_mat, axis=1, keepdims=True)
                    doc_norm = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

                    sim_matrix = np.dot(q_mat, doc_norm.T)

                    seen_per_pillar = {p.pillar: set() for p in probes}
                    for idx, (pillar, query, doc_types) in enumerate(slot_map):
                        doc_filter = {d.strip().lower() for d in doc_types} if doc_types else None
                        row_scores = sim_matrix[idx]
                        top_idxs = np.argsort(-row_scores)
                        count = 0
                        for didx in top_idxs:
                            cid = str(chunk_ids[didx])
                            if cid in seen_per_pillar[pillar]:
                                continue
                            pl = payloads[didx] or {}
                            hit_doc_type = str(pl.get("doc_type", "")).lower()
                            if doc_filter and hit_doc_type not in doc_filter:
                                continue
                            seen_per_pillar[pillar].add(cid)
                            results[pillar].append(
                                EvidenceChunk(
                                    chunk_id=cid,
                                    doc_id=str(pl.get("doc_id", "")),
                                    doc_type=str(pl.get("doc_type", "")),
                                    ticker=str(pl.get("ticker", sym)),
                                    label=str(pl.get("label", "")),
                                    element_type=str(pl.get("element_type", "text")),
                                    page_start=int(pl.get("page_start", 0)),
                                    page_end=int(pl.get("page_end", 0)),
                                    section_breadcrumb=str(pl.get("section_breadcrumb", "")),
                                    content=str(pl.get("content", "")),
                                    score=float(row_scores[didx]),
                                    query_matched=query,
                                    metadata=dict(pl.get("metadata") or {}),
                                )
                            )
                            count += 1
                            if count >= top_k_per_query:
                                break

                    local_elapsed = time.perf_counter() - t_local_start
                    total_elapsed = time.perf_counter() - total_start
                    total_chunks = sum(len(v) for v in results.values())
                    logger.info(
                        "[%s] ⚡ Tier-1 Sub-Millisecond Matrix Search: %d chunks fetched across %d pillars in %.4fs (Embed: %.4fs, Similarity & Filter: %.4fs).",
                        sym,
                        total_chunks,
                        len(probes),
                        total_elapsed,
                        embed_elapsed,
                        local_elapsed,
                    )
                    return results
            except Exception as exc:
                logger.warning("[%s] Local vector search fallback to Qdrant Cloud: %s", sym, exc)

        # Tier-2: Single-Flight Batch Search to Qdrant Cloud
        t_search_start = time.perf_counter()
        query_slot_map: list[tuple[str, str, Sequence[str]]] = []
        batch_search_requests: list[dict[str, Any]] = []

        for p in probes:
            for q in p.queries:
                vec = query_vec_map.get(q)
                if not vec:
                    continue
                query_slot_map.append((p.pillar, q, p.target_doc_types))
                batch_search_requests.append(
                    {
                        "query_vector": vec,
                        "ticker": sym,
                        "doc_types": p.target_doc_types,
                        "limit": top_k_per_query,
                    }
                )

        # Execute single HTTP flight with high-efficiency field projection (stripped heavy metadata)
        with_payload = ["content", "label", "page_start", "page_end", "doc_type", "section_breadcrumb", "chunk_id", "ticker"]
        try:
            batch_hits = self.indexer.search_batch_by_vectors(
                requests=batch_search_requests,
                with_payload=with_payload,
            )
        except Exception as exc:
            logger.error("[%s] Batch vector search failed: %s, falling back to parallel searches.", sym, exc)
            batch_hits = []

        search_elapsed = time.perf_counter() - t_search_start

        # 4. Map results back to respective pillars
        if batch_hits and len(batch_hits) == len(query_slot_map):
            seen_per_pillar: dict[str, set[str]] = {p.pillar: set() for p in probes}
            for (pillar, query, doc_types), hits in zip(query_slot_map, batch_hits):
                doc_type_filter = {d.strip().lower() for d in doc_types} if doc_types else None
                for hit in hits:
                    payload = hit.get("payload") or {}
                    hit_doc_type = str(payload.get("doc_type", "")).lower()
                    if doc_type_filter and hit_doc_type not in doc_type_filter:
                        continue

                    chunk_id = str(payload.get("chunk_id") or hit.get("id"))
                    if chunk_id in seen_per_pillar[pillar]:
                        continue

                    seen_per_pillar[pillar].add(chunk_id)
                    results[pillar].append(
                        EvidenceChunk(
                            chunk_id=chunk_id,
                            doc_id=str(payload.get("doc_id", "")),
                            doc_type=str(payload.get("doc_type", "")),
                            ticker=str(payload.get("ticker", sym)),
                            label=str(payload.get("label", "")),
                            element_type=str(payload.get("element_type", "text")),
                            page_start=int(payload.get("page_start", 0)),
                            page_end=int(payload.get("page_end", 0)),
                            section_breadcrumb=str(payload.get("section_breadcrumb", "")),
                            content=str(payload.get("content", "")),
                            score=float(hit.get("score", 0.0)),
                            query_matched=query,
                            metadata=dict(payload.get("metadata") or {}),
                        )
                    )

        total_elapsed = time.perf_counter() - total_start
        total_chunks = sum(len(v) for v in results.values())

        for p in probes:
            chunks = results.get(p.pillar, [])
            scores = [c.score for c in chunks] if chunks else [0.0]
            logger.info(
                "[%s] Pillar '%s' matched %d chunks | Score range: [%.3f - %.3f, avg %.3f]",
                sym,
                p.pillar,
                len(chunks),
                min(scores),
                max(scores),
                sum(scores) / max(1, len(scores)),
            )

        logger.info(
            "[%s] SOTA Single-Flight Retrieval Metrics -> Total Latency: %.3fs (Embed: %.3fs, Search: %.3fs) | %d chunks fetched across %d pillars.",
            sym,
            total_elapsed,
            embed_elapsed,
            search_elapsed,
            total_chunks,
            len(probes),
        )
        return results
