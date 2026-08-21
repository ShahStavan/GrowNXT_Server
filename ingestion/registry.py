"""Per-ticker ingestion registry.

One JSON file per ticker records everything the ingestion layer has done to
that company's filings, so that a second run does no work the first run has
already completed. It is the only durable state in the ingestion layer: the
PDFs, the parse caches, the chunk caches, and the vector files on disk are all
addressed from here, and anything not named here is treated as absent.

Staleness is decided by identity, not by timestamp. Each stage records the
fingerprint of the input it consumed:

* A document is re-downloaded when the catalogue's URL for it changes, because
  Screener publishes a new link when the issuer files a new document.
* A document is re-parsed when the downloaded bytes change (content hash) or
  the parser's version changes.
* A document is re-chunked when the parse output or the chunker's parameters
  change.
* A document is re-embedded when the chunk set changes (chunk fingerprint) or
  the embedding model changes. Re-embedding on a model change matters: vectors
  from two different models are not comparable, and silently mixing them in one
  index degrades retrieval in a way that leaves no trace in the output.

The registry is written atomically -- to a temporary file in the same directory,
then renamed -- so an interrupted run cannot leave a half-written state file
that would strand documents already on disk.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import datetime as _datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Optional

# The folding rule is owned by core.config so that a stock's directory name
# is identical whichever layer creates it. Re-exported here because the
# registry is the ingestion layer's own entry point for it.
from core.config import safe_ticker

__all__ = ["safe_ticker"]

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
REGISTRY_FILENAME: str = "registry.json"

# Sub-directories of data/<TICKER>/ addressed by the registry.
DOCUMENTS_DIR: str = "documents"
PARSED_DIR: str = "parsed"
CHUNKS_DIR: str = "chunks"
VECTORS_DIR: str = "vectors"
FINDINGS_DIR: str = "findings"

# Document classes. The catalogue exposes exactly these three.
DOC_TYPE_ANNUAL_REPORT: str = "annual_report"
DOC_TYPE_TRANSCRIPT: str = "concall_transcript"
DOC_TYPE_PRESENTATION: str = "concall_presentation"
DOC_TYPES: List[str] = [DOC_TYPE_ANNUAL_REPORT, DOC_TYPE_TRANSCRIPT, DOC_TYPE_PRESENTATION]

# Stage names, in pipeline order.
STAGES: List[str] = ["download", "parse", "chunk", "embed", "prompt"]


def utc_now() -> str:
    """Returns the current UTC time as an ISO-8601 string with a Z suffix."""
    return _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_text(text: str) -> str:
    """Returns the SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Returns the SHA-256 hex digest of a file, read in blocks.

    Args:
        path: File to hash.
        block_size: Read size in bytes. Annual reports run to tens of
            megabytes, so the file is never held in memory whole.

    Returns:
        Hex digest, or an empty string when the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with open(str(path), "rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                digest.update(block)
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return ""
    return digest.hexdigest()


@dataclass
class StageState:
    """The outcome of one pipeline stage for one document.

    Attributes:
        status: One of ``pending``, ``done``, or ``failed``.
        fingerprint: Identity of the input this stage consumed. The stage is
            stale when the current input's fingerprint differs.
        version: Version tag of the code that produced the output, so that a
            parser or chunker change invalidates its own output.
        at: UTC timestamp of the last attempt.
        detail: Stage-specific counters and paths.
        error: Message from the last failure, if any.
    """

    status: str = "pending"
    fingerprint: str = ""
    version: str = ""
    at: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def is_done(self) -> bool:
        """True when the stage completed successfully."""
        return self.status == "done"

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form, omitting an empty error for readability."""
        out: Dict[str, Any] = {
            "status": self.status,
            "fingerprint": self.fingerprint,
            "version": self.version,
            "at": self.at,
            "detail": self.detail,
        }
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "StageState":
        """Rebuilds a StageState from its JSON form."""
        data = payload or {}
        return cls(
            status=str(data.get("status", "pending")),
            fingerprint=str(data.get("fingerprint", "")),
            version=str(data.get("version", "")),
            at=str(data.get("at", "")),
            detail=dict(data.get("detail") or {}),
            error=str(data.get("error", "")),
        )


@dataclass
class DocumentRecord:
    """One catalogued document and its progress through the pipeline.

    Attributes:
        doc_id: Stable identifier, unique within a ticker. Derived from the
            document class and its period so that the same filing keeps the
            same identifier across runs.
        doc_type: One of DOC_TYPES.
        label: Human-readable period label, e.g. ``FY2026`` or ``Jul 2026``.
        source_url: URL the catalogue currently publishes for this document.
        period: Raw period fields as published by the catalogue.
        stages: Stage name to StageState.
        history: Append-only log of the events that changed this record.
    """

    doc_id: str
    doc_type: str
    label: str
    source_url: str
    period: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, StageState] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def stage(self, name: str) -> StageState:
        """Returns a stage's state, creating a pending one if absent."""
        if name not in self.stages:
            self.stages[name] = StageState()
        return self.stages[name]

    def record(self, event: str, **fields: Any) -> None:
        """Appends one event to the document's history."""
        entry: Dict[str, Any] = {"event": event, "at": utc_now()}
        entry.update({k: v for k, v in fields.items() if v not in (None, "")})
        self.history.append(entry)

    @property
    def url_fingerprint(self) -> str:
        """Identity of the catalogue link, which drives re-download."""
        return sha256_text(self.source_url)

    def relative_path(self, sub_dir: str, suffix: str) -> str:
        """Returns this document's path within the ticker directory."""
        return sub_dir + "/" + self.doc_id + suffix

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form of the record."""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "label": self.label,
            "source_url": self.source_url,
            "source_url_sha256": self.url_fingerprint,
            "period": self.period,
            "stages": {name: state.to_dict() for name, state in self.stages.items()},
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentRecord":
        """Rebuilds a DocumentRecord from its JSON form."""
        stages = {
            name: StageState.from_dict(value)
            for name, value in (payload.get("stages") or {}).items()
        }
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            doc_type=str(payload.get("doc_type", "")),
            label=str(payload.get("label", "")),
            source_url=str(payload.get("source_url", "")),
            period=dict(payload.get("period") or {}),
            stages=stages,
            history=list(payload.get("history") or []),
        )


class DocumentRegistry:
    """Reads, mutates, and writes one ticker's ingestion state.

    Args:
        ticker: Stock ticker symbol.
        data_dir: Root data directory. The registry lives at
            ``<data_dir>/<TICKER>/registry.json``.
    """

    def __init__(self, ticker: str, data_dir: Path) -> None:
        self.ticker = safe_ticker(ticker)
        self.root = Path(data_dir) / self.ticker
        self.path = self.root / REGISTRY_FILENAME
        self.company_name: str = ""
        self.catalog: Dict[str, Any] = {}
        self.documents: Dict[str, DocumentRecord] = {}
        # Ticker-level, not per document: the evidence dossier is a pivot across
        # every filing, so it has no single document to hang off.
        self.dossier: Dict[str, Any] = {}
        self.runs: List[Dict[str, Any]] = []
        self._load()

    # --- Persistence ---------------------------------------------------------

    def _load(self) -> None:
        """Loads the registry from disk, tolerating absence and corruption."""
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # A corrupt registry must not be silently discarded: losing it
            # would re-download every filing and re-embed every chunk.
            backup = self.path.with_suffix(".corrupt.json")
            logger.error("Registry at %s is unreadable (%s); preserved at %s",
                         self.path, exc, backup)
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return

        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            logger.warning("Registry schema %d differs from %d; re-deriving state.",
                           version, SCHEMA_VERSION)
        self.company_name = str(payload.get("company_name", ""))
        self.catalog = dict(payload.get("catalog") or {})
        self.dossier = dict(payload.get("dossier") or {})
        self.runs = list(payload.get("runs") or [])
        for doc_id, record in (payload.get("documents") or {}).items():
            self.documents[doc_id] = DocumentRecord.from_dict(record)

    def save(self) -> None:
        """Writes the registry to disk atomically."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "updated_at": utc_now(),
            "catalog": self.catalog,
            "dossier": self.dossier,
            "documents": {doc_id: rec.to_dict() for doc_id, rec in sorted(self.documents.items())},
            "runs": self.runs[-20:],
        }
        text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(self.root),
            prefix=".registry-", suffix=".tmp", delete=False,
        )
        try:
            with handle:
                handle.write(text)
            os.replace(handle.name, str(self.path))
        except OSError as exc:
            logger.error("Could not write registry %s: %s", self.path, exc)
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    # --- Paths ---------------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Returns an absolute path for a registry-relative path."""
        return self.root / relative

    def ensure_dirs(self) -> None:
        """Creates the per-ticker sub-directories the pipeline writes into."""
        for name in (DOCUMENTS_DIR, PARSED_DIR, CHUNKS_DIR, VECTORS_DIR, FINDINGS_DIR):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    # --- Documents -----------------------------------------------------------

    def upsert(
        self,
        doc_id: str,
        doc_type: str,
        label: str,
        source_url: str,
        period: Optional[Dict[str, Any]] = None,
    ) -> DocumentRecord:
        """Adds or refreshes one catalogued document.

        A changed URL is the signal that the issuer has published a new version
        of this document, so every stage downstream of the download is reset.
        Everything else about the record, including its history, is preserved.

        Args:
            doc_id: Stable identifier for the document.
            doc_type: One of DOC_TYPES.
            label: Human-readable period label.
            source_url: URL currently published by the catalogue.
            period: Raw period fields from the catalogue.

        Returns:
            The stored record.
        """
        existing = self.documents.get(doc_id)
        if existing is None:
            record = DocumentRecord(
                doc_id=doc_id,
                doc_type=doc_type,
                label=label,
                source_url=source_url,
                period=dict(period or {}),
            )
            record.record("catalogued", source_url=source_url)
            self.documents[doc_id] = record
            return record

        existing.label = label or existing.label
        existing.period = dict(period or existing.period)
        if existing.source_url != source_url:
            previous = existing.source_url
            existing.source_url = source_url
            existing.stages = {}
            existing.record("source_url_changed", previous_url=previous, source_url=source_url)
            logger.info("[%s] %s: catalogue link changed; will re-ingest.",
                        self.ticker, doc_id)
        return existing

    def by_type(self, doc_type: str) -> List[DocumentRecord]:
        """Returns the records of one document class, newest label first."""
        records = [r for r in self.documents.values() if r.doc_type == doc_type]
        return sorted(records, key=lambda r: r.doc_id, reverse=True)

    def select(self, doc_types: Optional[Iterable[str]] = None) -> List[DocumentRecord]:
        """Returns records of the given classes in a stable pipeline order."""
        wanted = list(doc_types) if doc_types else DOC_TYPES
        out: List[DocumentRecord] = []
        for doc_type in wanted:
            out.extend(self.by_type(doc_type))
        return out

    # --- Stage bookkeeping ---------------------------------------------------

    def needs(
        self,
        record: DocumentRecord,
        stage: str,
        fingerprint: str,
        version: str,
        force: bool = False,
    ) -> bool:
        """Decides whether a stage must run for one document.

        Args:
            record: Document under consideration.
            stage: Stage name from STAGES.
            fingerprint: Identity of the stage's current input.
            version: Version tag of the stage's implementation.
            force: Run regardless of recorded state.

        Returns:
            True when the stage has never completed, its input has changed, its
            implementation has changed, or an output file it claims to have
            written is missing.
        """
        if force:
            return True
        state = record.stage(stage)
        if not state.is_done:
            return True
        if state.fingerprint != fingerprint or state.version != version:
            return True
        for key in ("path", "vectors_path", "cache"):
            relative = state.detail.get(key)
            if relative and not self.resolve(str(relative)).exists():
                logger.info("[%s] %s: %s output %s is missing; rerunning stage.",
                            self.ticker, record.doc_id, stage, relative)
                return True
        return False

    def complete(
        self,
        record: DocumentRecord,
        stage: str,
        fingerprint: str,
        version: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Marks a stage done for one document."""
        state = record.stage(stage)
        state.status = "done"
        state.fingerprint = fingerprint
        state.version = version
        state.at = utc_now()
        state.detail = dict(detail or {})
        state.error = ""
        record.record(stage + "_done", **{
            k: v for k, v in (detail or {}).items()
            if isinstance(v, (str, int, float))
        })

    def fail(self, record: DocumentRecord, stage: str, error: str) -> None:
        """Marks a stage failed for one document, keeping the reason."""
        state = record.stage(stage)
        state.status = "failed"
        state.at = utc_now()
        state.error = str(error)[:500]
        record.record(stage + "_failed", error=state.error)
        logger.warning("[%s] %s: %s failed: %s", self.ticker, record.doc_id, stage, error)

    # --- Reporting -----------------------------------------------------------

    def add_run(self, summary: Dict[str, Any]) -> None:
        """Appends one run summary to the registry."""
        self.runs.append(summary)

    def status_table(self) -> List[Dict[str, Any]]:
        """Returns one row per document describing each stage's state."""
        rows: List[Dict[str, Any]] = []
        for record in self.select():
            row: Dict[str, Any] = {
                "doc_id": record.doc_id,
                "doc_type": record.doc_type,
                "label": record.label,
            }
            for stage in STAGES:
                row[stage] = record.stage(stage).status
            embed = record.stage("embed").detail
            row["embed_model"] = str(embed.get("model", ""))
            row["chunks"] = int(record.stage("chunk").detail.get("count", 0) or 0)
            rows.append(row)
        return rows

    def counts(self) -> Dict[str, int]:
        """Returns the number of documents that have completed each stage."""
        out: Dict[str, int] = {"documents": len(self.documents)}
        for stage in STAGES:
            out[stage] = sum(
                1 for record in self.documents.values() if record.stage(stage).is_done
            )
        return out
