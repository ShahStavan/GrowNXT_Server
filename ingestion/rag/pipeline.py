"""LangGraph Institutional Qualitative RAG Pipeline for GrowNXT.

Orchestrates the 5-node adaptive research workflow:
  inspect_and_plan -> parallel_retrieve -> rerank_evidence -> synthesize_findings -> format_and_persist

Persists institutional findings to output/<TICKER>/findings/findings.json and findings.md.
Google Python Style Guide Compliant.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = Any
    START = "START"
    END = "END"

from core.config import OUTPUT_DIR, safe_ticker
from ingestion.indexer import IndexerConfig, QdrantVectorIndexer
from ingestion.rag.probes import ThematicProbe, build_adaptive_probes
from ingestion.rag.reranker import EvidenceReranker
from ingestion.rag.retriever import EvidenceChunk, ParallelVectorRetriever
from ingestion.rag.synthesizer import (
    InstitutionalSynthesizer,
    ResearchDossier,
    ThematicFinding,
    utc_now,
)

logger = logging.getLogger(__name__)

FINDINGS_DIR_NAME: str = "findings"
FINDINGS_JSON_NAME: str = "findings.json"
FINDINGS_MD_NAME: str = "findings.md"


class ResearchState(TypedDict, total=False):
    """State carried through the LangGraph institutional research graph."""

    ticker: str
    company_name: str
    available_doc_types: List[str]
    probes: List[Dict[str, Any]]
    retrieved_evidence: Dict[str, List[Dict[str, Any]]]
    reranked_evidence: Dict[str, List[Dict[str, Any]]]
    findings: Dict[str, Dict[str, Any]]
    dossier: Dict[str, Any]
    typst_snippet: str
    markdown_snippet: str
    status: str
    error: Optional[str]
    elapsed_seconds: float


class InstitutionalRAGPipeline:
    """End-to-end adaptive RAG pipeline orchestrator."""

    def __init__(
        self,
        indexer: Optional[QdrantVectorIndexer] = None,
        config: Optional[IndexerConfig] = None,
    ) -> None:
        self.indexer = indexer or QdrantVectorIndexer(config=config)
        self.retriever = ParallelVectorRetriever(indexer=self.indexer)
        self.reranker = EvidenceReranker(top_k=5)
        self.synthesizer = InstitutionalSynthesizer()
        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    # --- Node 1: Inspect State & Plan Adaptive Probes ---
    def inspect_and_plan(self, state: ResearchState) -> ResearchState:
        """Inspects document availability and generates tailored thematic probes."""
        ticker = safe_ticker(state["ticker"])
        logger.info("[%s] Node 1: Inspecting document state and planning probes...", ticker)

        # Inspect ticker state from disk
        ticker_state = self.indexer.load_ticker_state(ticker)
        available_doc_types: set[str] = set()

        chunks_dir = self.indexer.config.output_dir / ticker / "chunks"
        if chunks_dir.exists():
            for f in chunks_dir.glob("*.json"):
                stem = f.stem.lower()
                if "annual" in stem or "report" in stem:
                    available_doc_types.add("annual_report")
                if "presentation" in stem:
                    available_doc_types.add("concall_presentation")
                    available_doc_types.add("presentation")
                if "transcript" in stem:
                    available_doc_types.add("concall_transcript")
                    available_doc_types.add("transcript")

        # If no documents are ingested or indexed yet, auto-ingest available filings from the catalogue
        if not available_doc_types and not ticker_state.documents:
            discovered = self._auto_ingest_and_index(ticker)
            if discovered:
                available_doc_types.update(discovered)
                ticker_state = self.indexer.load_ticker_state(ticker)

        doc_types_list = sorted(list(available_doc_types)) if available_doc_types else ["annual_report"]
        probes = build_adaptive_probes(ticker=ticker, available_doc_types=doc_types_list)

        logger.info(
            "[%s] Identified %d available doc types %s; generated %d adaptive probes.",
            ticker,
            len(doc_types_list),
            doc_types_list,
            len(probes),
        )

        return {
            **state,
            "available_doc_types": doc_types_list,
            "probes": [p.to_dict() for p in probes],
            "status": "PLANNING_COMPLETED",
        }

    def _auto_ingest_and_index(self, ticker: str, max_docs: int = 2) -> set[str]:
        """Automatically discovers, downloads, extracts, chunks, and indexes filings when absent."""
        found_types: set[str] = set()
        logger.info("[%s] No filings on disk or vector index; discovering filings from catalogue...", ticker)
        try:
            from ingestion.catalog import fetch_catalog
            from ingestion.documents.download import Downloader, DownloadRequest
            from ingestion.documents.storage import DocumentStore
            from ingestion.documents.extract import Extractor
            from ingestion.chunker import chunk_document, write_chunk_cache

            cat = fetch_catalog(ticker)
            if not cat.entries:
                logger.warning("[%s] No catalogue entries found upstream.", ticker)
                return found_types

            store = DocumentStore.open(ticker, data_dir=self.indexer.config.output_dir).ensure()
            downloader = Downloader(store=store)

            # Select: 1x Latest Annual Report, 3x Latest Concall Transcripts, 1x Latest Investor Presentation
            annual_entries = [e for e in cat.entries if e.doc_type == "annual_report"][:1]
            transcript_entries = [e for e in cat.entries if e.doc_type == "concall_transcript"][:3]
            presentation_entries = [e for e in cat.entries if e.doc_type == "concall_presentation"][:1]

            target_entries = annual_entries + transcript_entries + presentation_entries
            logger.info(
                "[%s] Selected %d filings for ingestion (annual=%d, transcripts=%d, presentations=%d)",
                ticker,
                len(target_entries),
                len(annual_entries),
                len(transcript_entries),
                len(presentation_entries),
            )

            requests = [
                DownloadRequest(doc_id=e.doc_id, url=e.source_url, label=e.label)
                for e in target_entries
            ]
            results = list(downloader.fetch_all(requests))

            extractor = Extractor(ocr=False, figures=False)
            for r in results:
                if r.ok and r.path:
                    entry = next((e for e in target_entries if e.doc_id == r.doc_id), None)
                    doc_type = entry.doc_type if entry else "concall_transcript"
                    try:
                        doc_extracted = extractor.run(
                            pdf=r.path,
                            store=store,
                            doc_id=r.doc_id,
                            doc_type=doc_type,
                            ticker=ticker,
                            label=r.doc_id,
                            write=True,
                        )
                        chunk_set = chunk_document(doc_extracted, chunk_size=800, chunk_overlap=100)
                        chunk_path = store.root / "chunks" / f"{r.doc_id}.json"
                        write_chunk_cache(chunk_set, chunk_path)
                        found_types.add(doc_type)
                        logger.info("[%s] Ingested and chunked %s (%d chunks).", ticker, r.doc_id, chunk_set.n_chunks)
                    except Exception as extract_err:
                        logger.warning("[%s] Failed to extract %s: %s; continuing with other filings.", ticker, r.doc_id, extract_err)

            # Index all successfully chunked documents into Qdrant & local matrix cache
            try:
                self.indexer.index_ticker_documents(ticker)
                logger.info("[%s] Indexed new filings into Qdrant vector store.", ticker)
            except Exception as index_err:
                logger.warning("[%s] Indexing into Qdrant encountered error: %s", ticker, index_err)
        except Exception as exc:
            logger.warning("[%s] Auto-ingestion failed: %s", ticker, exc)

        return found_types

    # --- Node 2: Parallel Vector Retrieval ---
    def parallel_retrieve(self, state: ResearchState) -> ResearchState:
        """Executes concurrent vector searches in Qdrant across all probes."""
        ticker = safe_ticker(state["ticker"])
        probes_data = state.get("probes") or []
        probes = [ThematicProbe(**p) for p in probes_data]

        logger.info("[%s] Node 2: Executing parallel vector retrieval...", ticker)
        raw_evidence_map = self.retriever.retrieve_all_probes_parallel(
            ticker=ticker,
            probes=probes,
            top_k_per_query=6,
        )

        serializable_evidence: dict[str, list[dict[str, Any]]] = {
            k: [c.to_dict() for c in v]
            for k, v in raw_evidence_map.items()
        }

        return {
            **state,
            "retrieved_evidence": serializable_evidence,
            "status": "RETRIEVAL_COMPLETED",
        }

    # --- Node 3: Evidence Reranking ---
    def rerank_evidence(self, state: ResearchState) -> ResearchState:
        """Filters and reranks raw candidate chunks for maximum precision."""
        ticker = safe_ticker(state["ticker"])
        logger.info("[%s] Node 3: Reranking and fusing candidate evidence...", ticker)

        raw_evidence_map = state.get("retrieved_evidence") or {}
        typed_map: dict[str, list[EvidenceChunk]] = {
            k: [EvidenceChunk.from_dict(c) for c in v]
            for k, v in raw_evidence_map.items()
        }

        reranked_map = self.reranker.rerank_all(typed_map, top_k=5)
        serializable_reranked: dict[str, list[dict[str, Any]]] = {
            k: [c.to_dict() for c in v]
            for k, v in reranked_map.items()
        }

        return {
            **state,
            "reranked_evidence": serializable_reranked,
            "status": "RERANKING_COMPLETED",
        }

    # --- Node 4: Institutional LLM Synthesis ---
    def synthesize_findings(self, state: ResearchState) -> ResearchState:
        """Generates grounded sell-side research findings for each pillar via hosted LLM."""
        ticker = safe_ticker(state["ticker"])
        logger.info("[%s] Node 4: Synthesizing institutional research findings...", ticker)

        probes_data = state.get("probes") or []
        probes = [ThematicProbe(**p) for p in probes_data]

        reranked_data = state.get("reranked_evidence") or {}
        reranked_map: dict[str, list[EvidenceChunk]] = {
            k: [EvidenceChunk.from_dict(c) for c in v]
            for k, v in reranked_data.items()
        }

        dossier = self.synthesizer.synthesize_all(
            ticker=ticker,
            reranked_evidence=reranked_map,
            probes=probes,
            company_name=state.get("company_name", ticker),
            available_doc_types=state.get("available_doc_types", []),
        )

        serializable_findings = {
            k: v.to_dict() for k, v in dossier.pillars.items()
        }

        return {
            **state,
            "findings": serializable_findings,
            "dossier": dossier.to_dict(),
            "status": "SYNTHESIS_COMPLETED",
        }

    # --- Node 5: Format & Persist Findings ---
    def format_and_persist(self, state: ResearchState) -> ResearchState:
        """Formats Typst and Markdown snippets and atomically writes findings.json to disk."""
        ticker = safe_ticker(state["ticker"])
        logger.info("[%s] Node 5: Formatting and persisting findings to disk...", ticker)

        dossier_data = state.get("dossier") or {}
        dossier = ResearchDossier.from_dict(dossier_data) if dossier_data else None

        # Build Markdown representation
        md_text = dossier.to_markdown() if dossier else ""

        # Build Typst representation
        typst_blocks: list[str] = [
            f"// --- Institutional Qualitative Evidence: {ticker} ---",
            "#block(sticky: true, above: 10pt, below: 5pt)[",
            f"  #text(size: 11pt, weight: \"bold\", fill: rgb(\"#002B49\"))[Institutional Research Findings & Strategic Highlights: {ticker}]",
            "]",
        ]

        if dossier:
            for pillar, finding in dossier.pillars.items():
                typst_blocks.append(f"\n#block(above: 7pt, below: 3pt)[#text(weight: \"semibold\", fill: rgb(\"#007A87\"))[{finding.title}]]")
                for bullet in finding.bullet_points:
                    # Escape raw dollar signs or hash symbols for Typst
                    esc_bullet = bullet.replace("$", "\\$").replace("#", "\\#")
                    typst_blocks.append(f"- {esc_bullet}")
                if finding.takeaway:
                    esc_takeaway = finding.takeaway.replace("$", "\\$").replace("#", "\\#")
                    typst_blocks.append(f"\n#text(style: \"italic\", fill: rgb(\"#333333\"))[💡 *Analyst Takeaway*: {esc_takeaway}]\n")

        typst_text = "\n".join(typst_blocks)

        # Persist to disk: output/<TICKER>/findings/findings.json
        findings_dir = self.indexer.config.output_dir / ticker / FINDINGS_DIR_NAME
        findings_dir.mkdir(parents=True, exist_ok=True)

        json_path = findings_dir / FINDINGS_JSON_NAME
        temp_json = json_path.with_suffix(f"{json_path.suffix}.tmp")
        temp_json.write_text(
            json.dumps(dossier_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(temp_json), str(json_path))

        md_path = findings_dir / FINDINGS_MD_NAME
        md_path.write_text(md_text, encoding="utf-8")

        logger.info(
            "[%s] Persisted findings to %s and %s.",
            ticker,
            json_path,
            md_path,
        )

        return {
            **state,
            "markdown_snippet": md_text,
            "typst_snippet": typst_text,
            "status": "COMPLETED",
        }

    def _build_graph(self) -> Any:
        """Constructs the LangGraph StateGraph pipeline."""
        if not LANGGRAPH_AVAILABLE:
            return None

        workflow = StateGraph(ResearchState)

        workflow.add_node("inspect_and_plan", self.inspect_and_plan)
        workflow.add_node("parallel_retrieve", self.parallel_retrieve)
        workflow.add_node("rerank_evidence", self.rerank_evidence)
        workflow.add_node("synthesize_findings", self.synthesize_findings)
        workflow.add_node("format_and_persist", self.format_and_persist)

        workflow.add_edge(START, "inspect_and_plan")
        workflow.add_edge("inspect_and_plan", "parallel_retrieve")
        workflow.add_edge("parallel_retrieve", "rerank_evidence")
        workflow.add_edge("rerank_evidence", "synthesize_findings")
        workflow.add_edge("synthesize_findings", "format_and_persist")
        workflow.add_edge("format_and_persist", END)

        return workflow.compile()

    def run(
        self,
        ticker: str,
        company_name: str = "",
        force: bool = False,
    ) -> ResearchState:
        """Executes the institutional RAG workflow for one ticker."""
        sym = safe_ticker(ticker)
        start_time = time.perf_counter()

        initial_state: ResearchState = {
            "ticker": sym,
            "company_name": company_name or sym,
            "available_doc_types": [],
            "probes": [],
            "retrieved_evidence": {},
            "reranked_evidence": {},
            "findings": {},
            "dossier": {},
            "typst_snippet": "",
            "markdown_snippet": "",
            "status": "INITIALIZED",
            "error": None,
            "elapsed_seconds": 0.0,
        }

        # Check existing findings cache if not force
        findings_path = self.indexer.config.output_dir / sym / FINDINGS_DIR_NAME / FINDINGS_JSON_NAME
        if findings_path.exists() and not force:
            try:
                data = json.loads(findings_path.read_text(encoding="utf-8"))
                dossier = ResearchDossier.from_dict(data)
                logger.info("[%s] Found cached findings at %s. Skipping re-generation.", sym, findings_path)
                return {
                    **initial_state,
                    "dossier": dossier.to_dict(),
                    "findings": {k: v.to_dict() for k, v in dossier.pillars.items()},
                    "markdown_snippet": dossier.to_markdown(),
                    "status": "CACHED",
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
            except Exception as exc:
                logger.warning("[%s] Could not read cached findings: %s. Re-running.", sym, exc)

        logger.info("[%s] Starting Institutional RAG Pipeline...", sym)

        if self._graph is not None:
            final_state = self._graph.invoke(initial_state)
        else:
            # Sequential execution fallback
            s1 = self.inspect_and_plan(initial_state)
            s2 = self.parallel_retrieve(s1)
            s3 = self.rerank_evidence(s2)
            s4 = self.synthesize_findings(s3)
            final_state = self.format_and_persist(s4)

        elapsed = time.perf_counter() - start_time
        final_state["elapsed_seconds"] = elapsed
        logger.info(
            "[%s] Completed Institutional RAG Pipeline in %.2fs (status=%s).",
            sym,
            elapsed,
            final_state.get("status"),
        )
        return final_state


def extract_ticker_findings(
    ticker: str,
    company_name: str = "",
    force: bool = False,
    config: Optional[IndexerConfig] = None,
) -> ResearchDossier:
    """Functional convenience API to extract institutional research findings."""
    pipeline = InstitutionalRAGPipeline(config=config)
    result_state = pipeline.run(ticker=ticker, company_name=company_name, force=force)
    dossier_data = result_state.get("dossier") or {}
    return ResearchDossier.from_dict(dossier_data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="GrowNXT Institutional Qualitative RAG Pipeline")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g. WIPRO, TCS, HDFCBANK)")
    parser.add_argument("--all", action="store_true", help="Run RAG pipeline across all stocks in output/")
    parser.add_argument("--force", action="store_true", help="Force re-generation ignoring cached findings")

    args = parser.parse_args()

    pipeline = InstitutionalRAGPipeline()

    if args.ticker:
        res = pipeline.run(ticker=args.ticker, force=args.force)
        print(f"\n==================== QUALITATIVE RESEARCH FINDINGS: {args.ticker} ====================")
        print(res.get("markdown_snippet", ""))
    elif args.all:
        output_root = pipeline.indexer.config.output_dir
        ticker_dirs = [d for d in output_root.iterdir() if d.is_dir()]
        for tdir in sorted(ticker_dirs):
            sym = tdir.name
            print(f"\n>>> Running RAG pipeline for {sym}...")
            res = pipeline.run(ticker=sym, force=args.force)
            print(f"[{sym}] Status: {res.get('status')} ({res.get('elapsed_seconds', 0):.2f}s)")
    else:
        parser.print_help()
