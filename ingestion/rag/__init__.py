"""Institutional Qualitative RAG layer for GrowNXT.

Extracts grounded sell-side research findings (Motilal Oswal, Kotak Equities style)
from Annual Reports, Investor Presentations, and Concall Transcripts via LangGraph.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from ingestion.rag.pipeline import (
    FINDINGS_DIR_NAME,
    FINDINGS_JSON_NAME,
    FINDINGS_MD_NAME,
    InstitutionalRAGPipeline,
    ResearchState,
    extract_ticker_findings,
)
from ingestion.rag.probes import (
    PILLAR_CAPITAL_ALLOCATION,
    PILLAR_CONCALL_HIGHLIGHTS,
    PILLAR_KEY_RISKS_AUDIT,
    PILLAR_MARGIN_COST,
    PILLAR_SEGMENT_DYNAMICS,
    PILLAR_STRATEGY_GROWTH,
    PILLAR_TITLES,
    ThematicProbe,
    build_adaptive_probes,
    get_default_probes,
)
from ingestion.rag.reranker import EvidenceReranker
from ingestion.rag.retriever import EvidenceChunk, ParallelVectorRetriever
from ingestion.rag.synthesizer import (
    InstitutionalSynthesizer,
    ResearchDossier,
    ThematicFinding,
)

__all__ = [
    "ThematicProbe",
    "EvidenceChunk",
    "ThematicFinding",
    "ResearchDossier",
    "ParallelVectorRetriever",
    "EvidenceReranker",
    "InstitutionalSynthesizer",
    "InstitutionalRAGPipeline",
    "ResearchState",
    "extract_ticker_findings",
    "build_adaptive_probes",
    "get_default_probes",
    "PILLAR_STRATEGY_GROWTH",
    "PILLAR_SEGMENT_DYNAMICS",
    "PILLAR_MARGIN_COST",
    "PILLAR_CAPITAL_ALLOCATION",
    "PILLAR_CONCALL_HIGHLIGHTS",
    "PILLAR_KEY_RISKS_AUDIT",
    "PILLAR_TITLES",
    "FINDINGS_DIR_NAME",
    "FINDINGS_JSON_NAME",
    "FINDINGS_MD_NAME",
]
