"""LangGraph ingestion pipeline for a ticker's filings.

Seven nodes, run in order, each idempotent:

    catalog -> plan -> fetch -> parse -> chunk -> embed -> prompts -> summary

The pipeline is expressed as a graph rather than a script for one reason that
matters operationally: every node's work is decided by the ``plan`` node from
the registry's recorded state, so a second run of the same graph does nothing.
That is the property the phase is verified against -- run twice for any ticker,
and the second run downloads nothing, re-parses nothing, and re-embeds nothing
-- and it only holds if each stage's decision is data, not control flow.

Stages fail per document, not per run. One filing whose host is down leaves its
own record marked failed with the reason, and the other filings complete; the
next run retries only what failed.

Google Python Style Guide Compliant.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - Python 3.7
    from typing_extensions import TypedDict  # type: ignore

from core.config import DATA_DIR
from ingestion.catalog import CatalogError, fetch_catalog
from ingestion.chunker import (
    CHUNKER_VERSION,
    DEFAULT_ANNUAL_SKIP_SECTIONS,
    chunk_document,
    read_chunk_cache,
    read_parse_cache,
    write_chunk_cache,
    write_parse_cache,
)
from ingestion.embedder import (
    DEFAULT_EMBED_MODEL,
    EMBEDDER_VERSION,
    EmbeddingClient,
    EmbeddingError,
    save_vectors,
)
from ingestion.dossier import DOSSIER_VERSION, build_dossier, write_dossier
from ingestion.fetcher import FETCHER_VERSION, download_document
from ingestion.parsers import parse_document, parser_version
from ingestion.prompts import PROMPT_VERSION, build_extraction_prompt, build_probes
from ingestion.registry import (
    CHUNKS_DIR,
    DOC_TYPES,
    DOCUMENTS_DIR,
    FINDINGS_DIR,
    PARSED_DIR,
    VECTORS_DIR,
    DocumentRecord,
    DocumentRegistry,
    sha256_text,
    utc_now,
)
from ingestion.retrieval import load_document_index, retrieve_for_probes

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = Any
    START = "START"
    END = "END"


class IngestionState(TypedDict, total=False):
    """State carried through the ingestion graph.

    Attributes:
        ticker: Stock ticker symbol.
        company_name: Name as published by the catalogue.
        planned: Document id to the list of stages that must run.
        actions: Stage name to the counts of what it did.
        errors: One entry per document that failed a stage.
        prompts: Built extraction prompts, keyed by document id.
        dossier: Summary of the evidence dossier written for this ticker.
        finished: True once the summary node has run.
    """

    ticker: str
    company_name: str
    planned: Dict[str, List[str]]
    actions: Dict[str, Dict[str, int]]
    errors: List[Dict[str, str]]
    prompts: Dict[str, Any]
    dossier: Dict[str, Any]
    finished: bool


class IngestionPipeline:
    """Runs the ingestion graph for one ticker.

    Args:
        ticker: Stock ticker symbol.
        data_dir: Root data directory. Defaults to the configured DATA_DIR.
        annual_reports: Annual report years to request from the catalogue.
        concall_years: Years of concalls to request from the catalogue.
        doc_types: Document classes to ingest. Defaults to all three.
        embed_model: Embedding model identifier.
        force: Stage names to run regardless of recorded state.
        build_prompts: Build extraction prompts after embedding.
        max_pages: Optional cap on pages parsed per document, for smoke tests.
        skip_sections: Annual-report sections to leave out of the chunk set.
            None applies DEFAULT_ANNUAL_SKIP_SECTIONS; an empty sequence keeps
            every section, which is how the whole volume gets embedded.
    """

    def __init__(
        self,
        ticker: str,
        data_dir: Optional[Path] = None,
        annual_reports: int = 1,
        concall_years: int = 1,
        doc_types: Optional[Sequence[str]] = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        force: Optional[Sequence[str]] = None,
        build_prompts: bool = True,
        max_pages: Optional[int] = None,
        skip_sections: Optional[Sequence[str]] = None,
    ) -> None:
        self.ticker = ticker.strip().upper()
        self.data_dir = Path(data_dir) if data_dir else Path(DATA_DIR)
        self.annual_reports = max(1, int(annual_reports))
        self.concall_years = max(0, int(concall_years))
        self.doc_types = list(doc_types) if doc_types else list(DOC_TYPES)
        self.embed_model = embed_model
        self.force = set(force or ())
        self.build_prompts = build_prompts
        self.max_pages = max_pages
        # None and [] mean different things here: the first asks for the
        # default, the second asks for nothing to be skipped.
        self.skip_sections = (
            list(DEFAULT_ANNUAL_SKIP_SECTIONS) if skip_sections is None
            else list(skip_sections)
        )

        self.registry = DocumentRegistry(self.ticker, self.data_dir)
        self.registry.ensure_dirs()
        self.client = EmbeddingClient(model=self.embed_model)
        self.started = time.time()

    # --- Helpers -------------------------------------------------------------

    def _records(self) -> List[DocumentRecord]:
        """Returns the records this run is allowed to touch."""
        return self.registry.select(self.doc_types)

    def _chunk_params(self) -> str:
        """Returns a fingerprint of the chunker configuration."""
        return sha256_text(json.dumps({
            "skip_sections": sorted(self.skip_sections),
            "version": CHUNKER_VERSION,
        }, sort_keys=True))

    @staticmethod
    def _bump(state: IngestionState, stage: str, key: str, amount: int = 1) -> None:
        """Increments one of a stage's action counters."""
        actions = state.setdefault("actions", {})
        bucket = actions.setdefault(stage, {})
        bucket[key] = bucket.get(key, 0) + amount

    @staticmethod
    def _note_error(state: IngestionState, doc_id: str, stage: str, message: str) -> None:
        """Records a per-document failure without aborting the run."""
        state.setdefault("errors", []).append(
            {"doc_id": doc_id, "stage": stage, "error": str(message)[:300]}
        )

    # --- Nodes ---------------------------------------------------------------

    def node_catalog(self, state: IngestionState) -> IngestionState:
        """Fetches the document catalogue and reconciles it with the registry."""
        try:
            catalog = fetch_catalog(
                self.ticker,
                annual_reports=self.annual_reports,
                concall_years=self.concall_years,
            )
        except CatalogError as exc:
            logger.error("[%s] catalogue unavailable: %s", self.ticker, exc)
            self._note_error(state, "-", "catalog", str(exc))
            state["company_name"] = self.registry.company_name
            return state

        self.registry.company_name = catalog.company_name or self.registry.company_name
        self.registry.catalog = catalog.meta
        for entry in catalog.entries:
            if entry.doc_type not in self.doc_types:
                continue
            self.registry.upsert(
                doc_id=entry.doc_id,
                doc_type=entry.doc_type,
                label=entry.label,
                source_url=entry.source_url,
                period=entry.period,
            )
            self._bump(state, "catalog", "catalogued")
        self.registry.save()
        state["company_name"] = self.registry.company_name
        return state

    def node_plan(self, state: IngestionState) -> IngestionState:
        """Decides which stages must run for which documents.

        This is the whole idempotency guarantee in one place: each stage's
        input fingerprint is compared with what the registry recorded last
        time, and only a difference schedules work.
        """
        planned: Dict[str, List[str]] = {}
        for record in self._records():
            stages: List[str] = []

            if self.registry.needs(record, "download", record.url_fingerprint,
                                   FETCHER_VERSION, force="download" in self.force):
                stages.append("download")

            download = record.stage("download")
            content_hash = str(download.detail.get("sha256", ""))
            parse_input = sha256_text(content_hash + "|" + str(self.max_pages))
            if "download" in stages or self.registry.needs(
                    record, "parse", parse_input, parser_version(record.doc_type),
                    force="parse" in self.force):
                stages.append("parse")

            parse_state = record.stage("parse")
            chunk_input = sha256_text(
                str(parse_state.fingerprint) + "|" + self._chunk_params())
            if "parse" in stages or self.registry.needs(
                    record, "chunk", chunk_input, CHUNKER_VERSION,
                    force="chunk" in self.force):
                stages.append("chunk")

            chunk_state = record.stage("chunk")
            embed_input = sha256_text(
                str(chunk_state.detail.get("fingerprint", "")) + "|" + self.embed_model)
            if "chunk" in stages or self.registry.needs(
                    record, "embed", embed_input, EMBEDDER_VERSION,
                    force="embed" in self.force):
                stages.append("embed")

            if stages:
                planned[record.doc_id] = stages

        state["planned"] = planned
        total = sum(len(stages) for stages in planned.values())
        if total:
            logger.info("[%s] plan: %d stage runs across %d documents",
                        self.ticker, total, len(planned))
            for doc_id, stages in sorted(planned.items()):
                logger.info("[%s]   %s: %s", self.ticker, doc_id, ", ".join(stages))
        else:
            logger.info("[%s] plan: nothing to do; every document is current.",
                        self.ticker)
        return state

    def node_fetch(self, state: IngestionState) -> IngestionState:
        """Downloads the documents the plan calls for."""
        planned = state.get("planned", {})
        for record in self._records():
            if "download" not in planned.get(record.doc_id, []):
                if record.stage("download").is_done:
                    self._bump(state, "download", "current")
                continue

            result = download_document(
                record,
                self.registry.resolve(DOCUMENTS_DIR),
                force="download" in self.force,
            )
            if not result.ok:
                self.registry.fail(record, "download", result.error)
                self._note_error(state, record.doc_id, "download", result.error)
                self._bump(state, "download", "failed")
                continue

            self.registry.complete(
                record, "download", record.url_fingerprint, FETCHER_VERSION,
                detail={
                    "path": result.relative_path,
                    "sha256": result.sha256,
                    "bytes": result.n_bytes,
                    "content_type": result.content_type,
                },
            )
            self._bump(state, "download", "reused" if result.reused else "downloaded")
        self.registry.save()
        return state

    def node_parse(self, state: IngestionState) -> IngestionState:
        """Parses the documents the plan calls for."""
        planned = state.get("planned", {})
        for record in self._records():
            if "parse" not in planned.get(record.doc_id, []):
                if record.stage("parse").is_done:
                    self._bump(state, "parse", "current")
                continue

            download = record.stage("download")
            if not download.is_done:
                self._bump(state, "parse", "skipped")
                continue

            path = self.registry.resolve(str(download.detail.get("path", "")))
            if not path.exists():
                self.registry.fail(record, "parse", "document missing at " + str(path))
                self._note_error(state, record.doc_id, "parse", "document missing")
                self._bump(state, "parse", "failed")
                continue

            try:
                document = parse_document(
                    path, record.doc_id, record.doc_type, self.ticker,
                    record.label, max_pages=self.max_pages,
                    company=self.registry.company_name,
                )
            except Exception as exc:  # noqa: BLE001 - one bad filing must not abort
                self.registry.fail(record, "parse", str(exc))
                self._note_error(state, record.doc_id, "parse", str(exc))
                self._bump(state, "parse", "failed")
                continue

            if not document.sections:
                message = "no text extracted"
                self.registry.fail(record, "parse", message)
                self._note_error(state, record.doc_id, "parse", message)
                self._bump(state, "parse", "failed")
                continue

            cache = self.registry.resolve(record.relative_path(PARSED_DIR, ".json"))
            write_parse_cache(document, cache)
            content_hash = str(download.detail.get("sha256", ""))
            self.registry.complete(
                record, "parse",
                sha256_text(content_hash + "|" + str(self.max_pages)),
                parser_version(record.doc_type),
                detail={
                    "cache": record.relative_path(PARSED_DIR, ".json"),
                    "pages": document.n_pages,
                    "chars": document.n_chars,
                    "blocks": document.n_blocks,
                    "sections": len(document.sections),
                    "section_sizes": document.section_sizes(),
                },
            )
            self._bump(state, "parse", "parsed")
            # Saved per document: parsing a 561-page annual report takes three
            # minutes, and an interruption must not discard the documents
            # already done.
            self.registry.save()
        self.registry.save()
        return state

    def node_chunk(self, state: IngestionState) -> IngestionState:
        """Chunks the documents the plan calls for."""
        planned = state.get("planned", {})
        for record in self._records():
            if "chunk" not in planned.get(record.doc_id, []):
                if record.stage("chunk").is_done:
                    self._bump(state, "chunk", "current")
                continue

            parse_state = record.stage("parse")
            if not parse_state.is_done:
                self._bump(state, "chunk", "skipped")
                continue

            cache = self.registry.resolve(str(parse_state.detail.get("cache", "")))
            document = read_parse_cache(cache) if cache.exists() else None
            if document is None:
                self.registry.fail(record, "chunk", "parse cache unreadable")
                self._note_error(state, record.doc_id, "chunk", "parse cache unreadable")
                self._bump(state, "chunk", "failed")
                continue

            skip = self.skip_sections if record.doc_type == "annual_report" else []
            chunk_set = chunk_document(document, skip_sections=skip)
            if not chunk_set.chunks:
                self.registry.fail(record, "chunk", "no chunks produced")
                self._note_error(state, record.doc_id, "chunk", "no chunks produced")
                self._bump(state, "chunk", "failed")
                continue

            target = self.registry.resolve(record.relative_path(CHUNKS_DIR, ".json"))
            write_chunk_cache(chunk_set, target)
            self.registry.complete(
                record, "chunk",
                sha256_text(str(parse_state.fingerprint) + "|" + self._chunk_params()),
                CHUNKER_VERSION,
                detail={
                    "cache": record.relative_path(CHUNKS_DIR, ".json"),
                    "count": len(chunk_set.chunks),
                    "fingerprint": chunk_set.fingerprint,
                    "section_counts": chunk_set.section_counts(),
                    "params": chunk_set.params,
                },
            )
            self._bump(state, "chunk", "chunked")
            self.registry.save()
        self.registry.save()
        return state

    def node_embed(self, state: IngestionState) -> IngestionState:
        """Embeds the chunk sets whose content has actually changed.

        The plan can only say that the chunk stage re-ran, not whether it
        produced anything different. That distinction is worth the extra check
        here because it is the difference between one document being re-embedded
        and eight: changing the annual-report section default re-chunks every
        document, but leaves the transcripts' chunks byte-identical, and
        re-embedding those buys nothing at the price of the whole API round
        trip. So the decision is made against the chunk set on disk, once it is
        loaded and its fingerprint is known.
        """
        planned = state.get("planned", {})
        for record in self._records():
            if "embed" not in planned.get(record.doc_id, []):
                if record.stage("embed").is_done:
                    self._bump(state, "embed", "current")
                continue

            chunk_state = record.stage("chunk")
            if not chunk_state.is_done:
                self._bump(state, "embed", "skipped")
                continue

            cache = self.registry.resolve(str(chunk_state.detail.get("cache", "")))
            chunk_set = read_chunk_cache(cache) if cache.exists() else None
            if chunk_set is None or not chunk_set.chunks:
                self.registry.fail(record, "embed", "chunk cache unreadable")
                self._note_error(state, record.doc_id, "embed", "chunk cache unreadable")
                self._bump(state, "embed", "failed")
                continue

            embed_input = sha256_text(chunk_set.fingerprint + "|" + self.embed_model)
            if not self.registry.needs(record, "embed", embed_input, EMBEDDER_VERSION,
                                       force="embed" in self.force):
                logger.info("[%s] chunks are unchanged; keeping existing vectors.",
                            record.doc_id)
                self._bump(state, "embed", "current")
                continue

            texts = [chunk.embed_text for chunk in chunk_set.chunks]
            logger.info("[%s] embedding %d chunks with %s",
                        record.doc_id, len(texts), self.embed_model)
            try:
                matrix, excluded, reason = self.client.embed_tolerating_rejections(
                    texts)
            except (EmbeddingError, RuntimeError) as exc:
                self.registry.fail(record, "embed", str(exc))
                self._note_error(state, record.doc_id, "embed", str(exc))
                self._bump(state, "embed", "failed")
                continue

            # The service applies a content filter, so a document can be
            # embeddable apart from a passage or two. Those are named in the
            # manifest and the rest of the filing is kept.
            refused = set(excluded)
            kept_ids = [
                chunk.chunk_id for index, chunk in enumerate(chunk_set.chunks)
                if index not in refused
            ]
            excluded_ids = [
                chunk_set.chunks[index].chunk_id for index in excluded
                if index < len(chunk_set.chunks)
            ]
            if not kept_ids:
                message = "every chunk was refused: " + (reason or "unknown")
                self.registry.fail(record, "embed", message)
                self._note_error(state, record.doc_id, "embed", message)
                self._bump(state, "embed", "failed")
                continue

            detail = save_vectors(
                self.registry.resolve(VECTORS_DIR),
                record.doc_id,
                self.embed_model,
                kept_ids,
                matrix,
                chunk_set.fingerprint,
                excluded_chunk_ids=excluded_ids,
                exclusion_reason=reason,
            )
            detail["chunk_fingerprint"] = chunk_set.fingerprint
            self.registry.complete(
                record, "embed", embed_input, EMBEDDER_VERSION, detail=detail,
            )
            self._bump(state, "embed", "embedded")
            self._bump(state, "embed", "vectors", len(kept_ids))
            if excluded_ids:
                self._bump(state, "embed", "refused_chunks", len(excluded_ids))
            self.registry.save()
        self.registry.save()
        return state

    def node_prompts(self, state: IngestionState) -> IngestionState:
        """Builds an extraction prompt per embedded document.

        Retrieval runs here rather than at report time so that the evidence a
        prompt was built from is recorded on disk beside it. A prompt whose
        evidence cannot be reconstructed cannot be audited when a figure in the
        finished report turns out to be wrong.
        """
        if not self.build_prompts:
            return state

        prompts: Dict[str, Any] = {}
        company = self.registry.company_name or self.ticker
        for record in self._records():
            embed_state = record.stage("embed")
            if not embed_state.is_done:
                continue

            # Rebuilding a prompt costs one embedding request for its probes,
            # so it is gated on the same identities as the stages before it.
            prompt_input = sha256_text("|".join([
                str(embed_state.detail.get("chunk_fingerprint", "")),
                self.embed_model,
                company,
            ]))
            if not self.registry.needs(record, "prompt", prompt_input, PROMPT_VERSION,
                                       force="prompt" in self.force):
                self._bump(state, "prompts", "current")
                continue

            document = load_document_index(
                self.registry.resolve(CHUNKS_DIR),
                self.registry.resolve(VECTORS_DIR),
                record.doc_id,
            )
            if document is None:
                self._bump(state, "prompts", "unavailable")
                continue

            probes = build_probes(
                record.doc_type, company, record.label,
                available_sections=document.sections,
            )
            if not probes:
                self._bump(state, "prompts", "no_probes")
                continue

            try:
                hits = retrieve_for_probes(document, probes, self.client)
            except (EmbeddingError, RuntimeError) as exc:
                self._note_error(state, record.doc_id, "prompts", str(exc))
                self._bump(state, "prompts", "failed")
                continue

            # Every probed target is listed in the prompt, including those whose
            # hits were deduplicated away by a higher-scoring probe. Dropping
            # them would leave the model neither asked about the target nor able
            # to report it as an absence, which is the one outcome that reads in
            # the finished report as though the target had been checked.
            covered = {hit.focus_id for hit in hits}
            built = build_extraction_prompt(
                ticker=self.ticker,
                company=company,
                doc_id=record.doc_id,
                doc_type=record.doc_type,
                label=record.label,
                chunks=[hit.chunk for hit in hits],
                focus_areas=[area for area, _q in probes],
                available_sections=document.sections,
            )
            built["focus_ids_with_evidence"] = sorted(covered)
            built["retrieved"] = [
                {"chunk_id": hit.chunk.chunk_id, "score": round(hit.score, 4),
                 "focus_id": hit.focus_id, "section": hit.chunk.section_id}
                for hit in hits
            ]
            built["company"] = company
            built["label"] = record.label
            built["built_at"] = utc_now()
            built["embed_model"] = self.embed_model

            target = self.registry.resolve(
                FINDINGS_DIR + "/" + record.doc_id + ".prompt.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(built, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            prompts[record.doc_id] = built
            self.registry.complete(
                record, "prompt", prompt_input, PROMPT_VERSION,
                detail={
                    "path": FINDINGS_DIR + "/" + record.doc_id + ".prompt.json",
                    "chunks_cited": len(built["chunk_ids"]),
                    "focus_ids": built["focus_ids"],
                    "prompt_chars": len(built["prompt"]),
                },
            )
            self._bump(state, "prompts", "built")

        state["prompts"] = prompts
        self.registry.save()
        return state

    def node_dossier(self, state: IngestionState) -> IngestionState:
        """Pivots the per-document evidence into the report's own sections.

        This is the hand-off to report generation. It is built from the prompt
        artifacts and chunk caches already on disk, so it costs no embedding
        calls and is reproducible from a completed ingest.
        """
        if not self.build_prompts:
            return state

        records = [r for r in self._records() if r.stage("prompt").is_done]
        if not records:
            self._bump(state, "dossier", "skipped")
            return state

        try:
            dossier = build_dossier(self.registry, records)
            detail = write_dossier(dossier, self.registry.resolve(FINDINGS_DIR))
        except Exception as exc:  # noqa: BLE001 - report rather than abort
            self._note_error(state, "-", "dossier", str(exc))
            self._bump(state, "dossier", "failed")
            return state

        previous = str(self.registry.dossier.get("fingerprint", ""))
        detail["dossier_version"] = DOSSIER_VERSION
        detail["built_at"] = utc_now()
        detail["documents"] = len(records)
        self.registry.dossier = detail
        state["dossier"] = detail
        self._bump(state, "dossier", "unchanged" if previous == detail["fingerprint"]
                   else "written")
        self.registry.save()
        return state

    def node_summary(self, state: IngestionState) -> IngestionState:
        """Writes the run summary into the registry."""
        summary = {
            "run_at": utc_now(),
            "seconds": round(time.time() - self.started, 1),
            "ticker": self.ticker,
            "doc_types": self.doc_types,
            "embed_model": self.embed_model,
            "requested": {
                "annual_reports": self.annual_reports,
                "concall_years": self.concall_years,
            },
            "forced": sorted(self.force),
            "actions": state.get("actions", {}),
            "errors": state.get("errors", []),
            "counts": self.registry.counts(),
            "dossier": state.get("dossier", {}),
        }
        self.registry.add_run(summary)
        self.registry.save()
        state["finished"] = True
        logger.info("[%s] run complete in %.1fs: %s", self.ticker,
                    summary["seconds"], json.dumps(summary["actions"]))
        return state

    # --- Graph ---------------------------------------------------------------

    NODES = [
        ("catalog", "node_catalog"),
        ("plan", "node_plan"),
        ("fetch", "node_fetch"),
        ("parse", "node_parse"),
        ("chunk", "node_chunk"),
        ("embed", "node_embed"),
        ("prompts", "node_prompts"),
        ("dossier", "node_dossier"),
        ("summary", "node_summary"),
    ]

    def build_graph(self) -> Any:
        """Compiles the LangGraph state machine."""
        workflow = StateGraph(IngestionState)
        for name, method in self.NODES:
            workflow.add_node(name, getattr(self, method))
        workflow.add_edge(START, self.NODES[0][0])
        for (left, _), (right, _) in zip(self.NODES, self.NODES[1:]):
            workflow.add_edge(left, right)
        workflow.add_edge(self.NODES[-1][0], END)
        return workflow.compile()

    def run(self) -> IngestionState:
        """Runs the pipeline once.

        Returns:
            The final state, including per-stage action counts and any
            per-document errors.
        """
        initial: IngestionState = {
            "ticker": self.ticker,
            "company_name": self.registry.company_name,
            "planned": {},
            "actions": {},
            "errors": [],
            "prompts": {},
            "dossier": {},
            "finished": False,
        }
        if LANGGRAPH_AVAILABLE:
            graph = self.build_graph()
            # The graph's default recursion limit counts node visits; this
            # pipeline is linear, so one visit per node is sufficient.
            result = graph.invoke(initial, {"recursion_limit": len(self.NODES) + 4})
            return dict(result)  # type: ignore[return-value]

        logger.warning("LangGraph is not installed; running the same nodes in order.")
        state = initial
        for _name, method in self.NODES:
            state = getattr(self, method)(state)
        return state


def ingest(
    ticker: str,
    **kwargs: Any,
) -> IngestionState:
    """Runs the ingestion pipeline for one ticker.

    Args:
        ticker: Stock ticker symbol.
        **kwargs: Passed through to IngestionPipeline.

    Returns:
        The pipeline's final state.
    """
    return IngestionPipeline(ticker, **kwargs).run()
