"""Document fetcher and record models for catalogued filings.

Downloads catalogued documents to ``output/<TICKER>/documents/<doc_id>.pdf``,
hashes them, and manages document records.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

FETCHER_VERSION: str = "fetcher/v1"

# Sub-directories of output/<TICKER>/
DOCUMENTS_DIR: str = "documents"
PARSED_DIR: str = "parsed"
CHUNKS_DIR: str = "chunks"
VECTORS_DIR: str = "vectors"
FINDINGS_DIR: str = "findings"

# Document classes. The catalogue exposes exactly these three.
DOC_TYPE_ANNUAL_REPORT: str = "annual_report"
DOC_TYPE_TRANSCRIPT: str = "concall_transcript"
DOC_TYPE_PRESENTATION: str = "concall_presentation"
DOC_TYPES: list[str] = [
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_TRANSCRIPT,
    DOC_TYPE_PRESENTATION,
]

# Stage names, in pipeline order.
STAGES: list[str] = ["download", "parse", "chunk", "embed", "prompt"]

# Exchange hosts reject non-browser agents outright.
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
}

PDF_MAGIC: bytes = b"%PDF"
MAX_BYTES: int = 200 * 1024 * 1024
MIN_BYTES: int = 8 * 1024


def utc_now() -> str:
    """Returns the current UTC time as an ISO-8601 string with a Z suffix."""
    return (
        _datetime.datetime.now(_datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_text(text: str) -> str:
    """Returns the SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Returns the SHA-256 hex digest of a file, read in blocks."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                digest.update(block)
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return ""
    return digest.hexdigest()


@dataclass
class StageState:
    """The outcome of one pipeline stage for one document."""

    status: str = "pending"
    fingerprint: str = ""
    version: str = ""
    at: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def is_done(self) -> bool:
        """True when the stage completed successfully."""
        return self.status == "done"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
    def from_dict(cls, payload: dict[str, Any] | None) -> StageState:
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
    """One catalogued document and its progress through the pipeline."""

    doc_id: str
    doc_type: str
    label: str
    source_url: str
    period: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, StageState] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def stage(self, name: str) -> StageState:
        if name not in self.stages:
            self.stages[name] = StageState()
        return self.stages[name]

    def record(self, event: str, **fields: Any) -> None:
        entry: dict[str, Any] = {"event": event, "at": utc_now()}
        entry.update({k: v for k, v in fields.items() if v not in (None, "")})
        self.history.append(entry)

    @property
    def url_fingerprint(self) -> str:
        return sha256_text(self.source_url)

    def relative_path(self, sub_dir: str, suffix: str) -> str:
        return sub_dir + "/" + self.doc_id + suffix

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, payload: dict[str, Any]) -> DocumentRecord:
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


@dataclass
class FetchResult:
    """Outcome of one download attempt."""

    ok: bool = False
    path: Path | None = None
    relative_path: str = ""
    sha256: str = ""
    n_bytes: int = 0
    reused: bool = False
    content_type: str = ""
    error: str = ""


def _request_headers(url: str) -> dict[str, str]:
    headers = dict(DOWNLOAD_HEADERS)
    try:
        parsed = urlparse(url)
    except ValueError:
        return headers
    if parsed.scheme in ("http", "https") and parsed.netloc:
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return headers


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError:
        return False


def download_document(
    record: DocumentRecord,
    destination: Path,
    timeout: int = 180,
    max_retries: int = 2,
    force: bool = False,
) -> FetchResult:
    """Downloads one catalogued document, unless it is already on disk."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (record.doc_id + ".pdf")
    relative = record.relative_path("documents", ".pdf")

    if target.exists() and not force:
        if _looks_like_pdf(target) and target.stat().st_size >= MIN_BYTES:
            return FetchResult(
                ok=True,
                path=target,
                relative_path=relative,
                sha256=sha256_file(target),
                n_bytes=target.stat().st_size,
                reused=True,
            )
        logger.warning(
            "[%s] existing file is not a usable PDF; re-downloading.", record.doc_id
        )

    last_error = ""
    for attempt in range(max_retries + 1):
        written = 0
        content_type = ""
        with tempfile.NamedTemporaryFile(
            dir=str(destination),
            prefix="." + record.doc_id + "-",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                with requests.get(
                    record.source_url,
                    headers=_request_headers(record.source_url),
                    timeout=timeout,
                    stream=True,
                    allow_redirects=True,
                ) as response:
                    content_type = str(response.headers.get("Content-Type", ""))
                    if not response.ok:
                        last_error = f"HTTP {response.status_code}"
                        raise OSError(last_error)
                    for block in response.iter_content(chunk_size=1 << 16):
                        if not block:
                            continue
                        written += len(block)
                        if written > MAX_BYTES:
                            raise OSError(f"response exceeds {MAX_BYTES} bytes")
                        handle.write(block)

                handle.close()

                if written < MIN_BYTES:
                    last_error = f"response too small ({written} bytes)"
                    raise OSError(last_error)
                if not _looks_like_pdf(temporary):
                    last_error = (
                        f"response is not a PDF (content-type {content_type!r})"
                    )
                    raise OSError(last_error)

                temporary.replace(target)
                digest = sha256_file(target)
                logger.info(
                    "[%s] downloaded %.1f MB", record.doc_id, written / (1024 * 1024)
                )
                return FetchResult(
                    ok=True,
                    path=target,
                    relative_path=relative,
                    sha256=digest,
                    n_bytes=written,
                    reused=False,
                    content_type=content_type,
                )

            except (requests.RequestException, OSError) as exc:
                last_error = str(exc) or last_error or "download failed"
                with contextlib.suppress(OSError):
                    handle.close()
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass
            logger.warning(
                "[%s] download failed (%s), attempt %d/%d",
                record.doc_id,
                last_error,
                attempt + 1,
                max_retries + 1,
            )
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))

    return FetchResult(ok=False, relative_path=relative, error=last_error)
