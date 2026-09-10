"""Concurrent, resumable download of catalogued filings into the store.

An ingest of one company pulls four to a dozen PDFs from unrelated hosts -- an
exchange attachment handler, the issuer's investor-relations CDN, sometimes a
registrar -- and any one of them can be slow or down. So downloads run on a
small thread pool and results are consumed **as they land**: a 40 MB annual
report behind a slow exchange host no longer holds up six transcripts that are
already available.

Four host behaviours shape the code, all observed rather than guessed:

* Exchange hosts refuse requests without a browser user agent, and want a
  referer from their own site. The referer is therefore derived per request from
  the URL being fetched, because the catalogue mixes hosts freely.
* Investor-relations sites answer a moved document with an HTML error page and a
  **200**. A response is accepted only once its bytes parse as a PDF, so an
  error page can never be filed as a filing.
* A 404 is not a 503. Permanent statuses are not retried; transient ones back
  off exponentially, honouring ``Retry-After`` when the host sends it.
* Hosts rate-limit per connection, so each worker thread keeps its own
  `Session` rather than sharing one `requests` does not promise to serialise.

Bytes stream to a temporary file **inside the destination directory** and are
renamed into place: an interrupted run leaves no truncated PDF for a later run
to mistake for a complete one, and the rename is atomic because it never crosses
a filesystem. The hash is computed from the stream, so a 40 MB filing is read
once rather than twice.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from ingestion.documents.storage import DocumentStore

logger = logging.getLogger(__name__)

# Bumped when a change here would produce a different file from the same URL.
DOWNLOAD_VERSION: str = "download/v1"

# A PDF declares itself in its first bytes. Readers tolerate a short preamble, so
# the magic is searched for in the opening kilobyte rather than demanded at zero.
PDF_MAGIC: bytes = b"%PDF"
MAGIC_WINDOW: int = 1024

# Below the floor a response is an error page or a stub, not a filing; above the
# ceiling it is a misdirected response -- the largest integrated annual reports
# seen run to roughly 60 MB.
MIN_BYTES: int = 8 * 1024
MAX_BYTES: int = 200 * 1024 * 1024

# Streaming read size: few hundred iterations for a 40 MB filing, still small
# enough that the size ceiling is enforced promptly.
STREAM_BLOCK: int = 1 << 16

# Modest on purpose: the aim is to stop one slow host holding up the rest, not to
# hammer an exchange server that rate-limits per connection anyway.
DEFAULT_WORKERS: int = 4

# (connect, read) timeouts: fail fast on an unreachable host, wait long for a
# large filing served slowly.
DEFAULT_TIMEOUT: tuple[float, float] = (15.0, 180.0)

DEFAULT_RETRIES: int = 2

# Capped so a rate-limited host cannot park a worker for minutes on the strength
# of its own Retry-After.
MAX_BACKOFF: float = 30.0

# Statuses worth another attempt. Everything else in 4xx is a property of the
# request, not of the moment.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
}


class _Transient(Exception):
    """The same request may well succeed on another attempt."""


class _Permanent(Exception):
    """The same request will fail the same way; retrying only wastes time."""


@dataclass(frozen=True)
class DownloadRequest:
    """One document to fetch.

    Deliberately not a registry record: the downloader has no opinion about
    pipeline state, which is what lets it be tested without one.

    Attributes:
        doc_id: Stable identifier; also the PDF's filename in the store.
        url: URL to fetch.
        label: Period label, used only to make logs readable.

    """

    doc_id: str
    url: str
    label: str = ""


@dataclass
class DownloadResult:
    """The outcome of one document's download.

    Attributes:
        doc_id: The document this describes.
        ok: True when a valid PDF is on disk at `path`.
        path: Absolute path to the stored PDF, when there is one.
        relative_path: Path relative to the stock directory, for the registry.
        sha256: Content hash of the stored bytes.
        n_bytes: Size on disk.
        n_pages: Page count from the PDF's own catalogue, or 0 when unreadable.
            Recorded so extraction can compare it against the pages it recovers
            and notice a silent partial failure.
        reused: True when a valid file was already present and not re-fetched.
        content_type: Content type the server reported.
        attempts: HTTP attempts made.
        seconds: Wall time spent on this document.
        error: Failure reason when `ok` is False.

    """

    doc_id: str
    ok: bool = False
    path: Path | None = None
    relative_path: str = ""
    sha256: str = ""
    n_bytes: int = 0
    n_pages: int = 0
    reused: bool = False
    content_type: str = ""
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def megabytes(self) -> float:
        """Size in MB, for logging."""
        return self.n_bytes / (1024.0 * 1024.0)


@dataclass
class DownloadBatch:
    """A running batch of downloads.

    Returned by `Downloader.start` so a caller can launch the fetches and get on
    with something else. `completed` drains them in the order they finish, not
    the order they were requested.

    Attributes:
        total: Documents submitted.

    """

    total: int
    _futures: list[Future[DownloadResult]] = field(default_factory=list, repr=False)
    _pool: ThreadPoolExecutor | None = field(default=None, repr=False)

    def completed(self) -> Iterator[DownloadResult]:
        """Yields results as they land, then releases the thread pool.

        A worker that raised is reported as a failed download rather than
        propagating, so one unavailable filing cannot abort an ingest of several.
        """
        try:
            for future in as_completed(self._futures):
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    logger.exception("A download worker failed outright.")
                    yield DownloadResult(doc_id="", error=str(exc) or "worker failed")
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=False)

    def wait(self) -> list[DownloadResult]:
        """Blocks until every download has finished and returns the results."""
        return list(self.completed())

    def cancel(self) -> int:
        """Cancels downloads that have not started; returns how many."""
        return sum(1 for future in self._futures if future.cancel())


class Downloader:
    """Fetches catalogued filings into a `DocumentStore`.

    Args:
        store: Destination store. Its directories are created on construction.
        workers: Concurrent downloads.
        timeout: (connect, read) timeout in seconds.
        retries: Additional attempts after a transient failure.
        verify: Read the stored PDF's page count as a structural check. Cheap --
            it parses the cross-reference table, not the content -- and it
            catches a truncated download that still begins with a PDF header.

    """

    def __init__(
        self,
        store: DocumentStore,
        workers: int = DEFAULT_WORKERS,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        verify: bool = True,
    ) -> None:
        self._store = store.ensure()
        self._workers = max(1, int(workers))
        self._timeout = timeout
        self._retries = max(0, int(retries))
        self._verify = verify
        # One Session per worker thread: requests does not promise a Session is
        # safe to share, and per-thread pooling keeps connections warm anyway.
        self._local = threading.local()

    # --- Public API ----------------------------------------------------------

    def fetch(self, request: DownloadRequest, force: bool = False) -> DownloadResult:
        """Downloads one document, unless a valid copy is already on disk.

        Args:
            request: What to fetch.
            force: Re-download even when a valid file is present.

        Returns:
            A result. Failures are reported, never raised.

        """
        started = time.monotonic()
        target = self._store.pdf(request.doc_id)

        if not force:
            reused = self._reuse(request.doc_id, target)
            if reused is not None:
                return reused

        def failure(error: str, attempts: int) -> DownloadResult:
            return DownloadResult(
                doc_id=request.doc_id,
                relative_path=self._store.relative(target),
                attempts=attempts,
                seconds=time.monotonic() - started,
                error=error,
            )

        last = "download failed"
        for attempt in range(1, self._retries + 2):
            try:
                sha256, content_type, n_bytes, n_pages = self._attempt(request, target)
            except _Permanent as exc:
                logger.error("[%s] %s -- not retrying.", request.doc_id, exc)
                return failure(str(exc), attempt)
            except _Transient as exc:
                last = str(exc)
                logger.warning(
                    "[%s] %s (attempt %d/%d)",
                    request.doc_id,
                    last,
                    attempt,
                    self._retries + 1,
                )
                if attempt <= self._retries:
                    time.sleep(self._backoff(attempt, exc))
                continue

            logger.info(
                "[%s] downloaded %.1f MB, %s page(s)",
                request.doc_id,
                n_bytes / (1024.0 * 1024.0),
                n_pages or "?",
            )
            return DownloadResult(
                doc_id=request.doc_id,
                ok=True,
                path=target,
                relative_path=self._store.relative(target),
                sha256=sha256,
                n_bytes=n_bytes,
                n_pages=n_pages,
                content_type=content_type,
                attempts=attempt,
                seconds=time.monotonic() - started,
            )

        return failure(last, self._retries + 1)

    def start(
        self,
        requests_: Sequence[DownloadRequest],
        force: bool = False,
    ) -> DownloadBatch:
        """Submits downloads to a thread pool and returns without waiting.

        Args:
            requests_: Documents to fetch.
            force: Re-download even when valid files are present.

        Returns:
            A batch whose `completed` yields results as they land.

        """
        pending = list(requests_)
        if not pending:
            return DownloadBatch(total=0)

        workers = min(self._workers, len(pending))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="download")
        futures = [pool.submit(self.fetch, request, force) for request in pending]
        logger.info(
            "[%s] fetching %d document(s) on %d worker(s)",
            self._store.ticker,
            len(pending),
            workers,
        )
        return DownloadBatch(total=len(pending), _futures=futures, _pool=pool)

    def fetch_all(
        self,
        requests_: Sequence[DownloadRequest],
        force: bool = False,
    ) -> Iterator[DownloadResult]:
        """Downloads several documents concurrently, yielding as they land."""
        return self.start(requests_, force=force).completed()

    def close(self) -> None:
        """Closes this thread's HTTP session."""
        session = getattr(self._local, "session", None)
        if session is not None:
            session.close()
            self._local.session = None

    def __enter__(self) -> Downloader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- Internals -----------------------------------------------------------

    @property
    def _session(self) -> requests.Session:
        """Returns this thread's session, building it on first use."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return session

    def _reuse(self, doc_id: str, target: Path) -> DownloadResult | None:
        """Returns a result for an existing valid file, or None to re-download.

        An annual report is a multi-megabyte request against an exchange server,
        so a re-run must not repeat it -- but a file left behind by an earlier
        failure must not be trusted either. Hence validation rather than a mere
        existence test.
        """
        if not target.exists():
            return None
        try:
            size = target.stat().st_size
        except OSError:
            return None
        if size < MIN_BYTES or not _has_pdf_magic(target):
            logger.warning(
                "[%s] file on disk is not a usable PDF; re-downloading.", doc_id
            )
            return None

        n_pages = _page_count(target) if self._verify else 0
        if self._verify and n_pages == 0:
            logger.warning(
                "[%s] file on disk has no readable pages; re-downloading.", doc_id
            )
            return None
        return DownloadResult(
            doc_id=doc_id,
            ok=True,
            path=target,
            relative_path=self._store.relative(target),
            sha256=_hash_file(target),
            n_bytes=size,
            n_pages=n_pages,
            reused=True,
        )

    def _attempt(
        self,
        request: DownloadRequest,
        target: Path,
    ) -> tuple[str, str, int, int]:
        """Makes one HTTP attempt and, on success, moves the file into place.

        Returns:
            (sha256, content_type, n_bytes, n_pages).

        Raises:
            _Transient: The attempt may succeed if repeated.
            _Permanent: The request itself is wrong; do not repeat it.

        """
        try:
            response = self._session.get(
                request.url,
                headers=_headers_for(request.url),
                timeout=self._timeout,
                stream=True,
                allow_redirects=True,
            )
        except requests.Timeout as exc:
            raise _Transient(f"timed out: {exc}") from exc
        except requests.RequestException as exc:
            raise _Transient(f"request failed: {exc}") from exc

        with response:
            if not response.ok:
                message = f"HTTP {response.status_code}"
                if response.status_code in RETRY_STATUSES:
                    raise _retry_after(message, response)
                raise _Permanent(message)

            content_type = str(response.headers.get("Content-Type", ""))
            declared = _declared_length(response)
            if declared is not None and declared > MAX_BYTES:
                raise _Permanent(
                    f"server declares {declared} bytes, over the {MAX_BYTES} ceiling"
                )

            temporary, sha256, n_bytes = self._stream(
                response, request.doc_id, target.parent
            )

        try:
            if n_bytes < MIN_BYTES:
                raise _Permanent(
                    f"response too small ({n_bytes} bytes); expected a filing"
                )
            if not _has_pdf_magic(temporary):
                # How investor-relations hosts answer a moved document: an HTML
                # error page served with a 200.
                raise _Permanent(
                    f"response is not a PDF (content-type {content_type!r})"
                )
            n_pages = _page_count(temporary) if self._verify else 0
            if self._verify and n_pages == 0:
                # Began with a PDF header but has no readable page tree: a
                # truncated transfer the magic-byte check cannot catch.
                raise _Transient("PDF has no readable pages; transfer looks truncated")
            temporary.replace(target)
        except Exception:
            _discard(temporary)
            raise

        return sha256, content_type, n_bytes, n_pages

    def _stream(
        self,
        response: requests.Response,
        doc_id: str,
        destination: Path,
    ) -> tuple[Path, str, int]:
        """Streams a response to a temporary file beside its destination.

        Hashing happens on the way past, so a 40 MB filing is read once. The
        temporary file shares the destination's directory, which is what makes
        the later rename atomic.

        Returns:
            (temporary path, sha256, bytes written).

        """
        destination.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        with tempfile.NamedTemporaryFile(
            dir=str(destination),
            prefix="." + doc_id + "-",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                for block in response.iter_content(chunk_size=STREAM_BLOCK):
                    if not block:
                        continue
                    written += len(block)
                    if written > MAX_BYTES:
                        raise _Permanent(
                            f"response exceeds the {MAX_BYTES} byte ceiling"
                        )
                    digest.update(block)
                    handle.write(block)
            except _Permanent:
                _discard(temporary)
                raise
            except (requests.RequestException, OSError) as exc:
                _discard(temporary)
                raise _Transient(f"transfer interrupted: {exc}") from exc
        return temporary, digest.hexdigest(), written

    def _backoff(self, attempt: int, exc: Exception) -> float:
        """Returns the seconds to wait before the next attempt."""
        requested = getattr(exc, "retry_after", 0.0)
        if requested:
            return min(float(requested), MAX_BACKOFF)
        return min(2.0 * attempt, MAX_BACKOFF)


# --- Module helpers -----------------------------------------------------------


def _headers_for(url: str) -> dict[str, str]:
    """Returns request headers carrying a referer from the URL's own host.

    An exchange attachment handler wants a referer from its own site or it
    redirects to the announcement page instead of serving the file. Deriving it
    per request keeps the downloader host-agnostic, which matters because the
    catalogue mixes exchange links, issuer CDNs and registrars freely.
    """
    headers = dict(BROWSER_HEADERS)
    try:
        parsed = urlparse(url)
    except ValueError:
        return headers
    if parsed.scheme in ("http", "https") and parsed.netloc:
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return headers


def _retry_after(message: str, response: requests.Response) -> _Transient:
    """Builds a transient error carrying the host's Retry-After, when sent."""
    error = _Transient(message)
    raw = str(response.headers.get("Retry-After", "")).strip()
    if raw.isdigit():
        error.retry_after = float(raw)
    return error


def _declared_length(response: requests.Response) -> int | None:
    """Returns the Content-Length the server declared, when it is usable."""
    raw = str(response.headers.get("Content-Length", "")).strip()
    return int(raw) if raw.isdigit() else None


def _has_pdf_magic(path: Path) -> bool:
    """Returns True when a file's opening kilobyte contains the PDF header."""
    try:
        with path.open("rb") as handle:
            return PDF_MAGIC in handle.read(MAGIC_WINDOW)
    except OSError:
        return False


def _page_count(path: Path) -> int:
    """Returns a PDF's page count, or 0 when it cannot be read.

    Parses the cross-reference table only, so this costs milliseconds even on a
    500-page filing. An encrypted-but-readable filing is accepted: several
    issuers apply an owner password that restricts printing while leaving the
    text extractable.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - declared in requirements
        logger.debug("pypdf is unavailable; skipping the page-count check.")
        return 0
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            with contextlib.suppress(Exception):  # an unreadable filing is caught below
                reader.decrypt("")
        return len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - a malformed PDF is a failed download
        logger.debug("Could not read the page count of %s: %s", path, exc)
        return 0


def _hash_file(path: Path, block: int = 1 << 20) -> str:
    """Returns a file's SHA-256, reading it in blocks."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(block), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return ""
    return digest.hexdigest()


def _discard(path: Path) -> None:
    """Removes a temporary file, ignoring an already-absent one."""
    with contextlib.suppress(OSError):
        path.unlink()
