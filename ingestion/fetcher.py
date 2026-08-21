"""Document downloader for catalogued filings.

Downloads one catalogued document to ``data/<TICKER>/documents/<doc_id>.pdf``,
hashes it, and reports what happened so the registry can record it. A document
already on disk whose catalogue link is unchanged is never fetched again -- an
annual report is a multi-megabyte request against an exchange server, and a
re-run of the pipeline must not repeat it.

Two classes of host serve these filings, and each needs care:

* Exchange hosts reject requests without a browser user agent, and serve their
  attachments through a redirect that must be followed. They also want a referer
  from their own site, so headers are derived per host rather than pinned to one
  exchange: the catalogue mixes exchange links with issuers' own CDNs freely.
* Issuer investor-relations sites return an HTML error page with a 200 status
  when a document has moved. A response is therefore accepted only if it
  actually begins with a PDF header, so that an error page can never be filed
  as a filing and parsed into nonsense.

Downloads go to a temporary file in the destination directory and are renamed
into place, so an interrupted run leaves no truncated PDF that a later run
would treat as complete.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

from ingestion.registry import DocumentRecord, sha256_file

logger = logging.getLogger(__name__)

FETCHER_VERSION: str = "fetcher/v1"

# Exchange hosts reject non-browser agents outright.
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
}


def _request_headers(url: str) -> Dict[str, str]:
    """Returns download headers appropriate to the URL's own host.

    The catalogue mixes hosts freely: the same company's filings arrive from an
    exchange's attachment handler and from its own investor-relations CDN, and a
    third issuer's from a registrar. An exchange attachment handler wants a
    referer from its own site or it redirects to the announcement page, so the
    referer is derived from the URL being fetched rather than pinned to one
    exchange -- sending an exchange's referer to an issuer's CDN is at best
    meaningless and at worst grounds for a refusal.

    Args:
        url: The URL about to be requested.

    Returns:
        Request headers, including a same-origin referer where one can be derived.
    """
    headers = dict(DOWNLOAD_HEADERS)
    try:
        parsed = urlparse(url)
    except ValueError:
        return headers
    if parsed.scheme in ("http", "https") and parsed.netloc:
        headers["Referer"] = "%s://%s/" % (parsed.scheme, parsed.netloc)
    return headers


PDF_MAGIC: bytes = b"%PDF"

# An annual report runs to tens of megabytes; anything far beyond that is a
# misdirected response rather than a filing.
MAX_BYTES: int = 200 * 1024 * 1024

# Below this, the response is an error page or a stub, not a filing.
MIN_BYTES: int = 8 * 1024


@dataclass
class FetchResult:
    """Outcome of one download attempt.

    Attributes:
        ok: True when a valid PDF is on disk at ``path``.
        path: Absolute path to the stored document, if any.
        relative_path: Path relative to the ticker directory, for the registry.
        sha256: Content hash of the stored bytes.
        n_bytes: Size on disk.
        reused: True when the file was already present and was not re-fetched.
        content_type: Content type reported by the server.
        error: Failure reason when ``ok`` is False.
    """

    ok: bool = False
    path: Optional[Path] = None
    relative_path: str = ""
    sha256: str = ""
    n_bytes: int = 0
    reused: bool = False
    content_type: str = ""
    error: str = ""


def _looks_like_pdf(path: Path) -> bool:
    """Returns True when a file begins with a PDF header."""
    try:
        with open(str(path), "rb") as handle:
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
    """Downloads one catalogued document, unless it is already on disk.

    Args:
        record: Registry record naming the URL and the target file name.
        destination: Directory to write into, normally
            ``data/<TICKER>/documents``.
        timeout: Per-request timeout in seconds.
        max_retries: Additional attempts for transient failures.
        force: Re-download even when a valid file is already present.

    Returns:
        A FetchResult. Failures are reported rather than raised so that one
        unavailable filing does not abort an ingest of several.
    """
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
        logger.warning("[%s] existing file is not a usable PDF; re-downloading.",
                       record.doc_id)

    last_error = ""
    for attempt in range(max_retries + 1):
        handle = tempfile.NamedTemporaryFile(
            dir=str(destination), prefix="." + record.doc_id + "-", suffix=".part",
            delete=False,
        )
        temporary = Path(handle.name)
        written = 0
        content_type = ""
        try:
            with requests.get(record.source_url, headers=_request_headers(record.source_url),
                              timeout=timeout, stream=True, allow_redirects=True) as response:
                content_type = str(response.headers.get("Content-Type", ""))
                if not response.ok:
                    last_error = "HTTP %d" % response.status_code
                    raise IOError(last_error)
                with handle:
                    for block in response.iter_content(chunk_size=1 << 16):
                        if not block:
                            continue
                        written += len(block)
                        if written > MAX_BYTES:
                            raise IOError("response exceeds %d bytes" % MAX_BYTES)
                        handle.write(block)

            if written < MIN_BYTES:
                last_error = "response too small (%d bytes)" % written
                raise IOError(last_error)
            if not _looks_like_pdf(temporary):
                # A 200 carrying an HTML error page is the common failure mode
                # for investor-relations hosts after a document is moved.
                last_error = "response is not a PDF (content-type %r)" % content_type
                raise IOError(last_error)

            os.replace(str(temporary), str(target))
            digest = sha256_file(target)
            logger.info("[%s] downloaded %.1f MB", record.doc_id, written / (1024 * 1024))
            return FetchResult(
                ok=True,
                path=target,
                relative_path=relative,
                sha256=digest,
                n_bytes=written,
                reused=False,
                content_type=content_type,
            )

        except (requests.RequestException, IOError, OSError) as exc:
            last_error = str(exc) or last_error or "download failed"
            try:
                handle.close()
            except OSError:
                pass
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
            logger.warning("[%s] download failed (%s), attempt %d/%d",
                           record.doc_id, last_error, attempt + 1, max_retries + 1)
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))

    return FetchResult(ok=False, relative_path=relative, error=last_error)
