"""The per-ticker ingestion stages, as independently callable functions.

`ingestion.batch` used to carry catalogue -> diff -> download -> extract ->
chunk -> index -> mirror inline, in one function, with closures mutating the
`TickerRun` being built. That shape had two costs: the stage boundaries were
only visible as comments, and nothing else could call a single stage.

Each stage here takes an explicit `StageContext`, returns a small result
object, and reports failure in that object rather than raising -- the same
convention `DownloadResult` and `IndexResult` already use. What a stage does
*not* do is decide what the failure means for the ticker: mapping a result to
a `TickerRun` status is orchestration, and lives with the orchestrator.

This module therefore knows about the catalogue, the document store, Docling,
the chunker and the indexer, and knows nothing about `TickerRun`,
`RunManifest`, `BatchOptions` or any graph runtime. `ingestion.batch` composes
these stages sequentially; a checkpointed graph can compose the same functions
with a durable boundary between any two of them.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.config import safe_ticker
from ingestion.fetcher import (
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
)
from ingestion.runlog import utc_now

if TYPE_CHECKING:
    from ingestion.catalog import Catalog, CatalogEntry
    from ingestion.chunker import ChunkSet
    from ingestion.documents.content import ExtractedDocument
    from ingestion.documents.download import DownloadResult
    from ingestion.documents.extract import Extractor
    from ingestion.documents.storage import DocumentStore
    from ingestion.indexer import (
        DocumentIndexState,
        IndexerConfig,
        QdrantVectorIndexer,
        TickerState,
    )
    from ingestion.nifty50 import Constituent
    from ingestion.runlog import RunLogger

logger = logging.getLogger(__name__)

VECTOR_CACHE_NAME: str = "vectors.npz"

# Why Layer 1 selected a document for processing. Re-exported by
# `ingestion.batch` for callers that have always read them from there.
REASON_NEW: str = "new"
REASON_RETRY: str = "retry_failed"
REASON_VERSION: str = "version_bump"
REASON_COLLECTION: str = "collection_change"
REASON_FORCED: str = "forced"


# --- Context -----------------------------------------------------------------


@dataclass(frozen=True)
class StageContext:
    """Everything the stages for one ticker share.

    Frozen because a context is an address plus settings, not accumulating
    state: two stages handed the same context must resolve the same paths and
    the same collection. Results carry what a stage produced.

    Attributes:
        symbol: Directory-safe ticker, as every artefact path uses it.
        raw_symbol: Exchange symbol as catalogued, for `fetch_catalog`.
        name: Company name from the constituent list, when known.
        forced: Whether Layer 1 is bypassed for this ticker.
        annual_reports: Newest annual reports to select.
        transcripts: Newest concall transcripts to select.
        presentations: Newest investor presentations to select.
        concall_years: Years of concalls to request from the catalogue.
        chunk_size: Chunker target size, in characters.
        chunk_overlap: Chunker overlap, in characters.
        fast_tables: TableFormer fast mode. Part of the extractor version.
        page_filter: Convert only an annual report's financial section.
            Also part of the extractor version, because a filtered
            extraction holds a fraction of the pages an unfiltered one
            does and must never be reused for it.
        device: Device override for Docling, or None to resolve.
        num_threads: Extraction threads, or None for physical cores.
        dry_run: Report what would happen without downloading or embedding. A
            dry run always fetches the catalogue, because reporting what is
            pending is the whole point of one.
        catalog_ttl_hours: How long a recorded catalogue check stays good for.
            Zero disables the skip.
        store: The ticker's on-disk workspace. An address; `download_pending`
            is what creates the directories.
        indexer: The vector store, which also owns `state.json`.
        runlog: Where stage boundaries, counters and timings are recorded.
    """

    symbol: str
    raw_symbol: str
    name: str
    forced: bool
    annual_reports: int
    transcripts: int
    presentations: int
    concall_years: int
    chunk_size: int
    chunk_overlap: int
    fast_tables: bool
    page_filter: bool
    device: str | None
    num_threads: int | None
    dry_run: bool
    catalog_ttl_hours: float
    store: DocumentStore
    indexer: QdrantVectorIndexer
    runlog: RunLogger

    @classmethod
    def build(
        cls,
        member: Constituent,
        options: Any,
        indexer: QdrantVectorIndexer,
        runlog: RunLogger,
    ) -> StageContext:
        """Resolves one ticker's context from a batch's options.

        Args:
            member: The constituent being processed.
            options: A `ingestion.batch.BatchOptions`. Taken as `Any` so this
                module stays independent of the batch layer; only the scalars
                named on this class are read from it.
            indexer: The vector store for this run.
            runlog: The run's logger.

        Returns:
            The context every stage for this ticker is called with.
        """
        from ingestion.documents.storage import DocumentStore

        symbol = safe_ticker(member.symbol)
        return cls(
            symbol=symbol,
            raw_symbol=member.symbol,
            name=member.name,
            forced=options.forced(member.symbol),
            annual_reports=options.annual_reports,
            transcripts=options.transcripts,
            presentations=options.presentations,
            concall_years=options.concall_years,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            fast_tables=options.fast_tables,
            page_filter=options.page_filter,
            device=options.device,
            num_threads=options.num_threads,
            dry_run=options.dry_run,
            catalog_ttl_hours=float(options.catalog_ttl_hours),
            store=DocumentStore.open(symbol, data_dir=options.output_root),
            indexer=indexer,
            runlog=runlog,
        )

    @property
    def collection(self) -> str:
        """The Qdrant collection this ticker's vectors belong in."""
        return self.indexer.config.collection_for(self.symbol)

    @property
    def selection(self) -> dict[str, int]:
        """The per-class document counts this run asks the catalogue for."""
        return {
            "annual_reports": int(self.annual_reports),
            "transcripts": int(self.transcripts),
            "presentations": int(self.presentations),
            "concall_years": int(self.concall_years),
        }

    @property
    def wanted_extract_version(self) -> str:
        """The extractor version a run with these settings would stamp."""
        from ingestion.documents.extract import extract_version

        return extract_version(
            accurate_tables=not self.fast_tables, page_filter=self.page_filter
        )


# --- Results -----------------------------------------------------------------


@dataclass
class CatalogueResult:
    """What one catalogue request produced, or why none was made.

    Attributes:
        ok: A usable answer was reached, by request or by skip.
        catalog: The fetched catalogue. None when `skipped` is set.
        entries_total: Entries the catalogue held.
        company_name: Issuer name the catalogue reported.
        no_documents: The issuer has published nothing this pipeline can use,
            which is a different outcome from a failed request.
        skipped: The request was not made, because this ticker's last check is
            still inside the TTL and every filing it found is embedded.
        known_documents: Doc ids already embedded, when `skipped` is set.
        error: Why the request failed.
        seconds: Wall time spent.
    """

    ok: bool = False
    catalog: Catalog | None = None
    entries_total: int = 0
    company_name: str = ""
    no_documents: bool = False
    skipped: bool = False
    known_documents: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


@dataclass
class DiffResult:
    """Layer 1's verdict for one ticker."""

    selected: list[CatalogEntry] = field(default_factory=list)
    pending: list[tuple[CatalogEntry, str]] = field(default_factory=list)
    unchanged: list[CatalogEntry] = field(default_factory=list)
    state: TickerState | None = None

    @property
    def pending_ids(self) -> set[str]:
        """Doc ids Layer 1 selected for processing."""
        return {entry.doc_id for entry, _ in self.pending}

    def chunk_count(self, doc_id: str) -> int:
        """Chunks `state.json` records for an unchanged document."""
        if self.state is None:
            return 0
        recorded = self.state.documents.get(doc_id)
        return recorded.chunk_count if recorded else 0


@dataclass
class DownloadStageResult:
    """Raw download results, in completion order, plus the stage's duration."""

    results: list[DownloadResult] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class ExtractResult:
    """One document's Docling pass."""

    ok: bool = False
    document: ExtractedDocument | None = None
    error: str = ""
    seconds: float = 0.0


@dataclass
class ChunkResult:
    """One document's chunking pass."""

    ok: bool = False
    chunk_set: ChunkSet | None = None
    error: str = ""
    seconds: float = 0.0


@dataclass
class IndexedDoc:
    """What Layer 2 did with one chunk file."""

    doc_id: str
    status: str = ""
    chunks_indexed: int = 0
    error: str = ""
    seconds: float = 0.0


@dataclass
class IndexStageResult:
    """Layer 2 over every chunk file on disk for one ticker."""

    documents: list[IndexedDoc] = field(default_factory=list)
    indexed_any: bool = False
    chunks_indexed: int = 0


@dataclass
class MirrorResult:
    """Qdrant's own point count, and whether the local mirror was rewritten."""

    points_total: int = 0
    mirrored: bool = False
    vectors: int = 0


# --- Layer 1: selection and diff ---------------------------------------------


def select_entries(
    catalog: Catalog,
    annual_reports: int = 1,
    transcripts: int = 1,
    presentations: int = 1,
) -> list[CatalogEntry]:
    """Picks the newest N documents of each class from a catalogue.

    `parse_catalog` orders entries newest first within each class, so the
    first N of a class are its most recent filings.
    """
    chosen: list[CatalogEntry] = []
    for doc_type, count in (
        (DOC_TYPE_ANNUAL_REPORT, annual_reports),
        (DOC_TYPE_TRANSCRIPT, transcripts),
        (DOC_TYPE_PRESENTATION, presentations),
    ):
        if count > 0:
            chosen.extend(catalog.of_type(doc_type)[:count])
    return chosen


def document_reason(
    recorded: DocumentIndexState | None,
    collection: str,
    extract_version: str,
) -> str | None:
    """Returns why a recorded filing needs re-processing, or None if current.

    The single definition of "current", used by Layer 1's per-entry diff and by
    the catalogue TTL's whole-ticker check. Two copies of this predicate is how
    `should_index_document` once ended up with an inverted final comparison
    that re-embedded every current document and skipped every real change.

    Args:
        recorded: What `state.json` holds for this doc id, or None.
        collection: The collection this run writes to.
        extract_version: The version this run's extractor will stamp.

    Returns:
        A `REASON_*` constant, or None when nothing needs doing.
    """
    from ingestion.chunker import CHUNKER_VERSION

    if recorded is None:
        return REASON_NEW
    if recorded.status != "INDEXED":
        return REASON_RETRY
    if recorded.chunker_version and recorded.chunker_version != CHUNKER_VERSION:
        return REASON_VERSION
    if recorded.extract_version and recorded.extract_version != extract_version:
        return REASON_VERSION
    if recorded.qdrant_collection != collection:
        return REASON_COLLECTION
    return None


def pending_reason(
    entry: CatalogEntry,
    state: TickerState,
    config: IndexerConfig,
    forced: bool = False,
    extract_version: str | None = None,
) -> str | None:
    """Returns why an entry needs processing, or None when it is already current.

    "Already known" means already *successfully* indexed by the current
    pipeline into the current collection -- not merely seen before -- so a
    transient failure or a chunker upgrade is never silently permanent.

    Args:
        entry: The catalogued filing.
        state: What this ticker's `state.json` records.
        config: Indexer configuration, for the collection this run writes to.
        forced: Bypass Layer 1 entirely.
        extract_version: Version this run's extractor will stamp. Defaults to
            the accurate-tables version. A run that changes the table mode
            passes its own, so documents extracted under the other mode are
            re-processed rather than left as a cache the run would not produce.
    """
    from ingestion.documents.extract import EXTRACT_VERSION

    if forced:
        return REASON_FORCED
    return document_reason(
        state.documents.get(entry.doc_id),
        config.collection_for(state.ticker),
        extract_version or EXTRACT_VERSION,
    )


def compute_pending(
    entries: Iterable[CatalogEntry],
    state: TickerState,
    config: IndexerConfig,
    forced: bool = False,
    extract_version: str | None = None,
) -> tuple[list[tuple[CatalogEntry, str]], list[CatalogEntry]]:
    """Splits entries into (pending with reason) and unchanged."""
    pending: list[tuple[CatalogEntry, str]] = []
    unchanged: list[CatalogEntry] = []
    for entry in entries:
        reason = pending_reason(
            entry, state, config, forced=forced, extract_version=extract_version
        )
        if reason:
            pending.append((entry, reason))
        else:
            unchanged.append(entry)
    return pending, unchanged


# --- Stages ------------------------------------------------------------------


def catalogue_skip(ctx: StageContext) -> CatalogueResult | None:
    """Returns a skip result when this ticker needs no catalogue request.

    Restarting a run that died at ticker forty otherwise spends thirty-nine
    catalogue requests, plus the inter-ticker delay each, proving what the last
    run already recorded. The skip is deliberately narrow: it applies only when
    every filing the last check found is embedded *by this configuration*, so a
    stale catalogue can never be the reason work is missed.

    Args:
        ctx: The ticker's stage context.

    Returns:
        The skip result, or None when a real request is required.
    """
    if ctx.catalog_ttl_hours <= 0 or ctx.forced or ctx.dry_run:
        return None

    state = ctx.indexer.load_ticker_state(ctx.symbol)
    if not state.documents or not state.catalog_checked_at:
        return None
    # A run that now wants more documents per class must look again, even
    # inside the window a narrower run could have skipped.
    if state.catalog_selection != ctx.selection:
        return None

    wanted_extract = ctx.wanted_extract_version
    collection = ctx.collection
    if any(
        document_reason(recorded, collection, wanted_extract) is not None
        for recorded in state.documents.values()
    ):
        return None

    age_hours = _age_hours(state.catalog_checked_at)
    if age_hours is None or age_hours > ctx.catalog_ttl_hours:
        return None

    known = sorted(state.documents)
    ctx.runlog.add("catalogues_skipped")
    ctx.runlog.event(
        "catalog_skipped",
        ticker=ctx.symbol,
        checked_at=state.catalog_checked_at,
        age_hours=round(age_hours, 2),
        ttl_hours=ctx.catalog_ttl_hours,
        known_documents=known,
    )
    logger.info(
        "[%s] catalogue checked %.1fh ago and all %d filing(s) are current; skipping.",
        ctx.symbol,
        age_hours,
        len(known),
    )
    return CatalogueResult(ok=True, skipped=True, known_documents=known)


def _age_hours(timestamp: str) -> float | None:
    """Returns how many hours ago an ISO-8601 timestamp was, or None if unusable."""
    try:
        recorded = _dt.datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=_dt.UTC)
    delta = _dt.datetime.now(_dt.UTC) - recorded
    return delta.total_seconds() / 3600.0


def fetch_catalogue(ctx: StageContext) -> CatalogueResult:
    """Fetches one ticker's document catalogue -- Layer 1's input.

    Never raises: a catalogue that cannot be read is one failed ticker, not a
    failed batch, and the distinction between "this issuer has published
    nothing we can use" and "the request failed" is carried on the result.

    Args:
        ctx: The ticker's stage context.

    Returns:
        The catalogue, a skip when `catalogue_skip` allows one, or a result
        explaining why there is neither.
    """
    from ingestion.catalog import CatalogError, fetch_catalog

    skipped = catalogue_skip(ctx)
    if skipped is not None:
        return skipped

    started = time.perf_counter()
    try:
        catalog = fetch_catalog(
            ctx.raw_symbol,
            annual_reports=ctx.annual_reports,
            concall_years=ctx.concall_years,
        )
    except CatalogError as exc:
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("catalog", seconds)
        empty = "no documents" in str(exc).lower()
        return CatalogueResult(
            no_documents=empty,
            error="" if empty else str(exc),
            seconds=seconds,
        )
    except Exception as exc:  # noqa: BLE001 - one ticker must not end the batch
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("catalog", seconds)
        return CatalogueResult(error=f"{type(exc).__name__}: {exc}", seconds=seconds)

    seconds = time.perf_counter() - started
    ctx.runlog.time_stage("catalog", seconds)
    ctx.runlog.add("documents_catalogued", len(catalog.entries))
    ctx.runlog.event(
        "catalog_fetched",
        ticker=ctx.symbol,
        entries_total=len(catalog.entries),
        by_doc_type={
            t: len(catalog.of_type(t))
            for t in (
                DOC_TYPE_ANNUAL_REPORT,
                DOC_TYPE_TRANSCRIPT,
                DOC_TYPE_PRESENTATION,
            )
        },
        company_name=catalog.company_name,
        screener_url=(catalog.meta or {}).get("screener_url"),
        elapsed_s=round(seconds, 3),
    )
    _record_catalogue_check(ctx)
    return CatalogueResult(
        ok=True,
        catalog=catalog,
        entries_total=len(catalog.entries),
        company_name=catalog.company_name,
        seconds=seconds,
    )


def _record_catalogue_check(ctx: StageContext) -> None:
    """Stamps `state.json` with when this catalogue was fetched, and for what.

    A best-effort write: the catalogue is already in hand, so failing to record
    the check must cost the next run a request, never this run its work.
    """
    try:
        state = ctx.indexer.load_ticker_state(ctx.symbol)
        state.catalog_checked_at = utc_now()
        state.catalog_selection = ctx.selection
        ctx.indexer.save_ticker_state(state)
    except Exception as exc:  # noqa: BLE001 - the TTL is an accelerator
        logger.warning("[%s] could not record the catalogue check: %s", ctx.symbol, exc)


def compute_diff(ctx: StageContext, catalog: Catalog) -> DiffResult:
    """Runs Layer 1: which of the newest filings this run must process.

    Args:
        ctx: The ticker's stage context.
        catalog: The catalogue `fetch_catalogue` returned.

    Returns:
        The selected entries split into pending (with a reason each) and
        unchanged, alongside the `state.json` the split was made against.
    """
    selected = select_entries(
        catalog,
        annual_reports=ctx.annual_reports,
        transcripts=ctx.transcripts,
        presentations=ctx.presentations,
    )
    ctx.runlog.add("documents_selected", len(selected))

    state = ctx.indexer.load_ticker_state(ctx.symbol)
    pending, unchanged = compute_pending(
        selected,
        state,
        ctx.indexer.config,
        ctx.forced,
        extract_version=ctx.wanted_extract_version,
    )
    ctx.runlog.add("documents_pending", len(pending))
    ctx.runlog.add("documents_unchanged", len(unchanged))

    reasons: dict[str, int] = {}
    for _entry, reason in pending:
        reasons[reason] = reasons.get(reason, 0) + 1
    ctx.runlog.event(
        "diff_computed",
        ticker=ctx.symbol,
        selected=len(selected),
        known=len(state.documents),
        pending=len(pending),
        unchanged=len(unchanged),
        pending_doc_ids=[e.doc_id for e, _ in pending],
        reason_counts=reasons,
    )
    return DiffResult(
        selected=selected, pending=pending, unchanged=unchanged, state=state
    )


def download_pending(
    ctx: StageContext, pending: Iterable[tuple[CatalogEntry, str]]
) -> DownloadStageResult:
    """Downloads every pending filing concurrently, reusing valid local copies.

    Creates the ticker's artefact directories: this is the first stage that
    writes, so naming a path earlier cannot leave empty directories behind.

    Args:
        ctx: The ticker's stage context.
        pending: Layer 1's pending entries, with their reasons.

    Returns:
        Every download result in completion order. Per-document outcomes are
        recorded by `note_download`, so that a document's download, extraction
        and chunking stay adjacent in the event stream.
    """
    from ingestion.documents.download import Downloader, DownloadRequest

    ctx.store.ensure()
    started = time.perf_counter()
    with Downloader(store=ctx.store) as downloader:
        requests_ = [
            DownloadRequest(doc_id=e.doc_id, url=e.source_url, label=e.label)
            for e, _ in pending
        ]
        results = list(downloader.fetch_all(requests_))
    seconds = time.perf_counter() - started
    ctx.runlog.time_stage("download", seconds)
    return DownloadStageResult(results=results, seconds=seconds)


def note_download(ctx: StageContext, result: DownloadResult) -> bool:
    """Records one download's outcome and says whether it can be extracted.

    Args:
        ctx: The ticker's stage context.
        result: One result from `download_pending`.

    Returns:
        True when a usable PDF is on disk.
    """
    ctx.runlog.event(
        "download_completed",
        ticker=ctx.symbol,
        doc_id=result.doc_id,
        ok=result.ok,
        reused=result.reused,
        bytes=result.n_bytes,
        pages=result.n_pages,
        http_attempts=result.attempts,
        elapsed_s=round(result.seconds, 3),
        error=result.error,
    )
    if not result.ok or not result.path:
        ctx.runlog.add("documents_download_failed")
        return False
    ctx.runlog.add(
        "documents_reused_from_cache" if result.reused else "documents_downloaded"
    )
    ctx.runlog.add("bytes_downloaded", 0 if result.reused else result.n_bytes)
    return True


def build_extractor(ctx: StageContext) -> Extractor:
    """Builds the Docling extractor this run's documents share.

    Constructing one is cheap -- the converter and its models are built on
    first use -- but sharing it across a ticker's documents is what keeps the
    models loaded once rather than once per filing.
    """
    from ingestion.documents.extract import Extractor

    return Extractor(
        ocr=False,
        figures=False,
        accurate_tables=not ctx.fast_tables,
        page_filter=ctx.page_filter,
        device=ctx.device,
        num_threads=ctx.num_threads,
    )


def extract_one(
    ctx: StageContext,
    extractor: Extractor,
    entry: CatalogEntry,
    pdf: Path,
    source_sha256: str = "",
) -> ExtractResult:
    """Runs Docling over one filing, reusing a matching cached extraction.

    Args:
        ctx: The ticker's stage context.
        extractor: The shared extractor from `build_extractor`.
        entry: The catalogued filing being extracted.
        pdf: The downloaded PDF.
        source_sha256: The PDF's digest, which the download already computed.
            Passing it is what keeps the cache check from re-reading a
            multi-megabyte filing once per run.

    Returns:
        The extracted document, or the failure recorded against it.
    """
    started = time.perf_counter()
    try:
        document = extractor.run(
            pdf=pdf,
            store=ctx.store,
            doc_id=entry.doc_id,
            doc_type=entry.doc_type,
            ticker=ctx.symbol,
            label=entry.label,
            write=True,
            source_sha256=source_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - recorded against the document
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("extract", seconds)
        ctx.runlog.add("documents_extract_failed")
        ctx.runlog.event(
            "extract_failed",
            ticker=ctx.symbol,
            doc_id=entry.doc_id,
            error=str(exc)[:500],
            elapsed_s=round(seconds, 3),
        )
        return ExtractResult(error=f"{type(exc).__name__}: {exc}", seconds=seconds)

    seconds = time.perf_counter() - started
    ctx.runlog.time_stage("extract", seconds)
    ctx.runlog.add("documents_extracted")
    ctx.runlog.add("pages_extracted", document.n_pages)
    ctx.runlog.event(
        "extract_completed",
        ticker=ctx.symbol,
        doc_id=entry.doc_id,
        pages=document.n_pages,
        source_pages=document.n_source_pages,
        blocks=len(document.blocks),
        tables=len(document.tables),
        figures=len(document.figures),
        extractor=document.extractor,
        elapsed_s=round(seconds, 3),
    )
    return ExtractResult(ok=True, document=document, seconds=seconds)


def chunk_one(
    ctx: StageContext,
    entry: CatalogEntry,
    document: ExtractedDocument,
) -> ChunkResult:
    """Chunks one extracted filing and writes its chunk cache.

    Args:
        ctx: The ticker's stage context.
        entry: The catalogued filing being chunked.
        document: The extraction from `extract_one`.

    Returns:
        The chunk set, or the failure recorded against the document.
    """
    from ingestion.chunker import (
        chunk_document,
        chunk_params,
        read_chunk_cache,
        write_chunk_cache,
    )

    started = time.perf_counter()
    path = ctx.store.chunk_file(entry.doc_id)
    wanted = chunk_params(
        document, chunk_size=ctx.chunk_size, chunk_overlap=ctx.chunk_overlap
    )
    cached = read_chunk_cache(path) if path.exists() else None
    if cached is not None and cached.chunks and cached.params == wanted:
        # Same extraction, same settings: the chunk file on disk is what this
        # call would write. `params` is compared whole rather than field by
        # field so that adding a setting cannot silently bypass the check.
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("chunk", seconds)
        ctx.runlog.add("documents_chunks_reused")
        logger.info(
            "[%s] reusing cached chunks for %s (%d chunks)",
            ctx.symbol,
            entry.doc_id,
            cached.n_chunks,
        )
        return ChunkResult(ok=True, chunk_set=cached, seconds=seconds)

    try:
        chunk_set = chunk_document(
            document,
            chunk_size=ctx.chunk_size,
            chunk_overlap=ctx.chunk_overlap,
        )
        write_chunk_cache(chunk_set, path)
    except Exception as exc:  # noqa: BLE001 - recorded against the document
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("chunk", seconds)
        ctx.runlog.add("documents_chunk_failed")
        return ChunkResult(error=f"{type(exc).__name__}: {exc}", seconds=seconds)

    seconds = time.perf_counter() - started
    ctx.runlog.time_stage("chunk", seconds)
    ctx.runlog.add("documents_chunked")
    ctx.runlog.add("chunks_total", chunk_set.n_chunks)
    ctx.runlog.event(
        "chunk_completed",
        ticker=ctx.symbol,
        doc_id=entry.doc_id,
        n_chunks=chunk_set.n_chunks,
        n_chars=chunk_set.n_chars,
        element_counts=chunk_set.element_counts(),
        fingerprint=chunk_set.fingerprint,
        chunker_version=(chunk_set.params or {}).get("version", ""),
        elapsed_s=round(seconds, 3),
    )
    return ChunkResult(ok=True, chunk_set=chunk_set, seconds=seconds)


def index_chunk_files(
    ctx: StageContext, pending_ids: set[str] | None = None
) -> IndexStageResult:
    """Runs Layer 2 over every chunk file on disk for this ticker.

    A cheap no-op for documents whose fingerprint already matches what
    `state.json` records, and also what turns a document that failed to embed
    on an earlier run back to INDEXED -- which is why it scans the directory
    rather than only the documents this run produced.

    Args:
        ctx: The ticker's stage context.
        pending_ids: Doc ids this run itself selected. On a forced ticker
            these bypass the fingerprint check; a chunk file the run did not
            touch is never forced, because re-embedding a stale one would
            record it as current.

    Returns:
        One entry per chunk file, in filename order.
    """
    selected = pending_ids or set()
    chunks_dir = ctx.store.chunks
    chunk_files = sorted(chunks_dir.glob("*.json")) if chunks_dir.exists() else []
    outcome = IndexStageResult()

    for chunk_file in chunk_files:
        doc_id = chunk_file.stem
        started = time.perf_counter()
        result = ctx.indexer.index_chunk_file(
            chunk_file, force=ctx.forced and doc_id in selected
        )
        seconds = time.perf_counter() - started
        ctx.runlog.time_stage("index", seconds)
        rate = round(result.chunks_indexed / seconds, 1) if seconds > 0 else 0.0
        ctx.runlog.event(
            "index_completed",
            ticker=ctx.symbol,
            doc_id=doc_id,
            status=result.status,
            chunks_indexed=result.chunks_indexed,
            fingerprint=result.fingerprint,
            model=ctx.indexer.config.model_alias,
            vector_dim=(
                ctx.indexer.vector_size if result.status == "INDEXED" else None
            ),
            collection=ctx.collection,
            chunks_per_s=rate,
            elapsed_s=round(seconds, 3),
            error=result.error or "",
        )
        if result.status == "INDEXED":
            outcome.indexed_any = True
            outcome.chunks_indexed += result.chunks_indexed
            ctx.runlog.add("documents_embedded")
            ctx.runlog.add("chunks_embedded", result.chunks_indexed)
            ctx.runlog.add("vectors_upserted", result.chunks_indexed)
        elif result.status == "FAILED":
            ctx.runlog.add("documents_index_failed")
        else:
            ctx.runlog.add("documents_index_skipped")

        outcome.documents.append(
            IndexedDoc(
                doc_id=doc_id,
                status=result.status,
                chunks_indexed=result.chunks_indexed,
                error=result.error or "",
                seconds=seconds,
            )
        )
    return outcome


def verify_and_mirror(ctx: StageContext, indexed_any: bool) -> MirrorResult:
    """Counts what Qdrant actually holds, then refreshes the local mirror.

    The count is verification rather than a stage: a failure here is reported
    and does not fail the ticker, because the vectors were already
    acknowledged by the server before this runs. The mirror
    (`vectors.npz` + `payloads.json`) is the retriever's fast path, so it is
    also rewritten when it is missing but Qdrant holds points -- that is how a
    run interrupted between embedding and mirroring heals itself.

    Args:
        ctx: The ticker's stage context.
        indexed_any: Whether this run embedded anything for this ticker.

    Returns:
        Qdrant's point count and whether the mirror was rewritten.
    """
    outcome = MirrorResult()
    try:
        outcome.points_total = ctx.indexer.count_points(ctx.symbol)
        ctx.runlog.event(
            "qdrant_verified",
            ticker=ctx.symbol,
            collection=ctx.collection,
            points_total=outcome.points_total,
            persistent=ctx.indexer.is_persistent,
        )
    except Exception as exc:  # noqa: BLE001 - verification, not a stage
        ctx.runlog.event(
            "qdrant_verify_failed", ticker=ctx.symbol, error=str(exc)[:500]
        )

    cache_missing = not (ctx.store.root / VECTOR_CACHE_NAME).exists()
    if not (indexed_any or (cache_missing and outcome.points_total > 0)):
        return outcome

    started = time.perf_counter()
    try:
        path, rows = ctx.indexer.write_local_vector_cache(ctx.symbol)
    except Exception as exc:  # noqa: BLE001 - the cache is an accelerator
        ctx.runlog.event("local_cache_failed", ticker=ctx.symbol, error=str(exc)[:500])
        return outcome

    ctx.runlog.add("local_caches_written")
    ctx.runlog.event(
        "local_cache_written",
        ticker=ctx.symbol,
        path=str(path),
        vectors=rows,
        bytes=path.stat().st_size if path.exists() else 0,
        elapsed_s=round(time.perf_counter() - started, 3),
    )
    outcome.mirrored = True
    outcome.vectors = rows
    return outcome
