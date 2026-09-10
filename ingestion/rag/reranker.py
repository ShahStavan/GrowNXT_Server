"""Evidence Reranker & Context Fusion Engine for GrowNXT RAG.

Reranks and filters candidate chunks using semantic density, section relevance,
and financial entity presence to select the highest-fidelity evidence.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ingestion.rag.probes import (
    PILLAR_CAPITAL_ALLOCATION,
    PILLAR_CONCALL_HIGHLIGHTS,
    PILLAR_KEY_RISKS_AUDIT,
    PILLAR_MARGIN_COST,
    PILLAR_SEGMENT_DYNAMICS,
    PILLAR_STRATEGY_GROWTH,
)
from ingestion.rag.retriever import EvidenceChunk

logger = logging.getLogger(__name__)

DEFAULT_FINAL_TOP_K: int = 5

# Pillar-specific high-value section keywords
SECTION_KEYWORDS: dict[str, list[str]] = {
    PILLAR_STRATEGY_GROWTH: [
        "strategy",
        "vision",
        "outlook",
        "growth",
        "roadmap",
        "capex",
        "investment",
        "future",
        "opportunity",
    ],
    PILLAR_SEGMENT_DYNAMICS: [
        "segment",
        "geography",
        "vertical",
        "business",
        "division",
        "market share",
        "revenue by",
        "products",
    ],
    PILLAR_MARGIN_COST: [
        "margin",
        "profit",
        "ebitda",
        "ebit",
        "cost",
        "raw material",
        "pricing",
        "inflation",
        "operating expense",
    ],
    PILLAR_CAPITAL_ALLOCATION: [
        "cash flow",
        "balance sheet",
        "debt",
        "working capital",
        "dividend",
        "liquidity",
        "borrowings",
        "payout",
    ],
    PILLAR_CONCALL_HIGHLIGHTS: [
        "conference call",
        "earnings",
        "transcript",
        "management",
        "q&a",
        "analyst",
        "discussion",
        "outlook",
    ],
    PILLAR_KEY_RISKS_AUDIT: [
        "audit",
        "risk",
        "internal control",
        "contingent",
        "legal",
        "compliance",
        "matter",
        "emphasis",
    ],
}

# Regex to detect financial numbers and metrics
RE_FINANCIAL_METRICS = re.compile(
    r"(?:₹|rs\.?|inr|\$|€|%|crore|cr|lakh|bps|cagr|ebitda|ebit|capex|fy\d{2,4}|q[1-4]fy\d{2,4})",
    re.IGNORECASE,
)

RE_TRANSCRIPT_NOISE = re.compile(
    r"(?:moderator:|ladies and gentlemen|good day and welcome|listen-only mode|opportunity to ask questions|safe harbor statement|press \*|participant lines|thank you for taking my question|all lines will be in|conference call will be recorded)",
    re.IGNORECASE,
)


class EvidenceReranker:
    """Reranks and filters raw candidate evidence chunks for high precision."""

    def __init__(self, top_k: int = DEFAULT_FINAL_TOP_K) -> None:
        self.top_k = top_k

    def _score_chunk(self, chunk: EvidenceChunk, pillar: str) -> float:
        """Computes a composite relevance score combining dense score and structural heuristics."""
        base_score = chunk.score
        content_text = chunk.content or ""

        # 1. Section Header Relevance Boost (+0.05 to +0.15)
        section_text = (chunk.section_breadcrumb or "").lower()
        target_keywords = SECTION_KEYWORDS.get(pillar, [])
        sec_matches = sum(1 for kw in target_keywords if kw in section_text)
        section_boost = min(0.15, sec_matches * 0.05)

        # 2. Financial Numerical & Metric Density Boost (+0.02 to +0.12)
        metric_matches = len(RE_FINANCIAL_METRICS.findall(content_text))
        metric_boost = min(0.12, metric_matches * 0.02)

        # 3. Structured Table & Figure Content Boost (+0.05 for tables in quantitative pillars)
        element_boost = 0.0
        if chunk.element_type == "table" and pillar in (
            PILLAR_MARGIN_COST,
            PILLAR_CAPITAL_ALLOCATION,
            PILLAR_SEGMENT_DYNAMICS,
        ):
            element_boost = 0.05

        # 4. Transcript / Moderator Noise Penalty (-0.35)
        noise_penalty = -0.35 if RE_TRANSCRIPT_NOISE.search(content_text) else 0.0

        # 5. Length Penalty for excessively short / snippet chunks (< 40 chars)
        length_penalty = -0.15 if len(content_text.strip()) < 40 else 0.0

        return (
            base_score
            + section_boost
            + metric_boost
            + element_boost
            + noise_penalty
            + length_penalty
        )

    def rerank_pillar_evidence(
        self,
        pillar: str,
        candidates: Sequence[EvidenceChunk],
        top_k: int | None = None,
    ) -> list[EvidenceChunk]:
        """Deduplicates and selects top ranked evidence chunks for a single research pillar."""
        if not candidates:
            return []

        limit = top_k or self.top_k

        # Deduplication by chunk_id and content hash
        unique_candidates: list[EvidenceChunk] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()

        for c in candidates:
            norm_content = " ".join(
                c.content.strip().split()[:20]
            )  # first 20 words signature
            if c.chunk_id in seen_ids or norm_content in seen_texts:
                continue
            seen_ids.add(c.chunk_id)
            seen_texts.add(norm_content)
            unique_candidates.append(c)

        # Score and rank
        scored_pairs = [
            (self._score_chunk(chunk=c, pillar=pillar), c) for c in unique_candidates
        ]
        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)

        return [pair[1] for pair in scored_pairs[:limit]]

    def rerank_all(
        self,
        evidence_map: dict[str, Sequence[EvidenceChunk]],
        top_k: int | None = None,
    ) -> dict[str, list[EvidenceChunk]]:
        """Reranks candidate evidence across all pillars."""
        reranked: dict[str, list[EvidenceChunk]] = {}
        for pillar, candidates in evidence_map.items():
            reranked[pillar] = self.rerank_pillar_evidence(
                pillar=pillar,
                candidates=candidates,
                top_k=top_k,
            )
        return reranked
