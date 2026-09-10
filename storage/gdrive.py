"""Google Drive delivery for generated PDF reports.

Uploaded once, link-shared, and tracked in a ``drive.json`` sidecar keyed by
the PDF's content hash, so an unchanged report is never re-uploaded. Written
against the REST API with ``requests`` rather than the official client, which
would add httplib2, protobuf and uritemplate to move one 700 KB file -- and
this way the whole path is testable without credentials.
"""

import contextlib
import datetime as _datetime
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from core.config import PROJECT_ROOT, safe_ticker, stock_dir

logger = logging.getLogger(__name__)

TOKEN_URL: str = "https://oauth2.googleapis.com/token"
API_BASE: str = "https://www.googleapis.com/drive/v3"
UPLOAD_BASE: str = "https://www.googleapis.com/upload/drive/v3"

# Narrowest scope that can create and manage the files this app itself uploads.
SCOPE: str = "https://www.googleapis.com/auth/drive.file"

DEFAULT_TOKEN_FILE: Path = PROJECT_ROOT / ".gdrive_token.json"
SIDECAR_NAME: str = "drive.json"
REQUEST_TIMEOUT: int = 120

# Refresh a little early rather than racing the expiry on a slow upload.
_EXPIRY_MARGIN_SECONDS: int = 120

# An OAuth refresh token for a real account, not a service account: a service
# account has no storage quota and cannot own files, only write into a Shared
# Drive, which needs Workspace. Access tokens are refreshed automatically;
# re-consent cannot be, which is what DriveAuthError reports.
_REAUTH_HINT: str = (
    "Re-authorise with: venv/Scripts/python.exe -m scripts.gdrive_auth . "
    "If this recurs weekly, publish the OAuth consent screen to Production -- "
    "refresh tokens issued by an app in Testing status expire after 7 days."
)


class DriveError(RuntimeError):
    """Raised when a Drive operation fails."""


class DriveAuthError(DriveError):
    """Raised when the stored grant no longer works and a human must re-consent."""


@dataclass(frozen=True)
class DriveFile:
    """A report as it exists in Drive.

    Attributes:
        file_id: Drive file id.
        name: File name in Drive.
        view_link: Drive's own viewer page.
        preview_link: Embeddable viewer, for an ``iframe``.
        direct_link: Direct download URL.
        shared: Whether link-sharing is in place.
        uploaded_at: UTC ISO-8601 timestamp of the upload.

    """

    file_id: str
    name: str
    view_link: str
    preview_link: str
    direct_link: str
    shared: bool
    uploaded_at: str

    def as_dict(self) -> dict[str, Any]:
        """Returns the JSON-serialisable form used by the API and the sidecar."""
        return {
            "file_id": self.file_id,
            "name": self.name,
            "view_link": self.view_link,
            "preview_link": self.preview_link,
            "direct_link": self.direct_link,
            "shared": self.shared,
            "uploaded_at": self.uploaded_at,
        }


def _now_iso() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _token_file() -> Path:
    """Returns the path the refresh token is read from and written back to."""
    configured = os.getenv("GDRIVE_TOKEN_FILE", "").strip()
    return Path(configured) if configured else DEFAULT_TOKEN_FILE


def _stored_token() -> str | None:
    """Reads the refresh token from the token file, if one is present."""
    path = _token_file()
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return (json.load(handle) or {}).get("refresh_token") or None
    except (OSError, ValueError) as exc:
        logger.warning("Could not read Drive token file %s: %s", path, exc)
        return None


def save_refresh_token(refresh_token: str) -> Path:
    """Persists a refresh token so the server survives a restart.

    Args:
        refresh_token: Token minted by the consent flow.

    Returns:
        Path: The file written.

    """
    path = _token_file()
    payload = {"refresh_token": refresh_token, "saved_at": _now_iso()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with contextlib.suppress(OSError):  # pragma: no cover - best effort on Windows
        path.chmod(0o600)
    return path


def credentials_present() -> bool:
    """Reports whether enough configuration exists to talk to Drive."""
    return bool(
        os.getenv("GDRIVE_CLIENT_ID", "").strip()
        and os.getenv("GDRIVE_CLIENT_SECRET", "").strip()
        and (os.getenv("GDRIVE_REFRESH_TOKEN", "").strip() or _stored_token())
    )


class DriveStore:
    """Uploads and shares files in one Google Drive account.

    Args:
        client_id: OAuth client id. Defaults to ``GDRIVE_CLIENT_ID``.
        client_secret: OAuth client secret. Defaults to ``GDRIVE_CLIENT_SECRET``.
        refresh_token: Long-lived grant. Defaults to the token file, then
            ``GDRIVE_REFRESH_TOKEN``.
        folder_id: Destination folder. Defaults to ``GDRIVE_FOLDER_ID``; when
            unset, files land in the account's root.
        session: HTTP session, injectable for tests.

    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        folder_id: str | None = None,
        session: Any | None = None,
    ) -> None:
        self.client_id = client_id or os.getenv("GDRIVE_CLIENT_ID", "").strip()
        self.client_secret = (
            client_secret or os.getenv("GDRIVE_CLIENT_SECRET", "").strip()
        )
        self.refresh_token = (
            refresh_token
            or _stored_token()
            or os.getenv("GDRIVE_REFRESH_TOKEN", "").strip()
        )
        self.folder_id = (
            folder_id
            if folder_id is not None
            else os.getenv("GDRIVE_FOLDER_ID", "").strip()
        )
        self.session = session or requests.Session()

        if not (self.client_id and self.client_secret and self.refresh_token):
            raise DriveAuthError(
                "Google Drive is not configured. Set GDRIVE_CLIENT_ID, "
                "GDRIVE_CLIENT_SECRET and a refresh token. " + _REAUTH_HINT
            )

        self._access_token: str | None = None
        self._expires_at: float = 0.0

    # -- authentication ----------------------------------------------------

    def _refresh_access_token(self) -> str:
        """Exchanges the refresh token for an access token.

        Returns:
            str: A valid bearer token.

        Raises:
            DriveAuthError: If the grant is no longer valid.
            DriveError: On a transport failure.

        """
        try:
            response = self.session.post(
                TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise DriveError(f"Token refresh request failed: {exc}") from exc

        body = _json_or_empty(response)
        if response.status_code >= 400:
            error = str(body.get("error", "")).lower()
            detail = body.get("error_description") or response.text[:200]
            # invalid_grant is terminal: the token was revoked, or expired with
            # the consent screen left in Testing status.
            if error in ("invalid_grant", "unauthorized_client", "invalid_client"):
                raise DriveAuthError(
                    f"Google refused the stored grant ({error}: {detail}). {_REAUTH_HINT}"
                )
            raise DriveError(
                f"Token refresh failed (HTTP {response.status_code}): {detail}"
            )

        token = body.get("access_token")
        if not token:
            raise DriveError("Token refresh returned no access token.")

        # Google may hand back a new refresh token; persisting it keeps the
        # next restart working.
        rotated = body.get("refresh_token")
        if rotated and rotated != self.refresh_token:
            self.refresh_token = rotated
            try:
                logger.info(
                    "Drive refresh token rotated; persisted to %s",
                    save_refresh_token(rotated),
                )
            except OSError as exc:
                logger.warning("Could not persist rotated Drive refresh token: %s", exc)

        self._access_token = str(token)
        expires_in = max(0, int(body.get("expires_in", 3600)))
        self._expires_at = time.time() + expires_in - _EXPIRY_MARGIN_SECONDS
        logger.debug("Drive access token refreshed; valid for %ds.", expires_in)
        return self._access_token

    def access_token(self) -> str:
        """Returns a valid access token, refreshing it when due."""
        if self._access_token and time.time() < self._expires_at:
            return self._access_token
        return self._refresh_access_token()

    def _call(
        self, method: str, url: str, retry_auth: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Issues an authorised Drive API call.

        A 401 is retried once against a freshly minted token, which covers an
        access token that lapsed mid-session.
        """
        headers = dict(kwargs.pop("headers", None) or {})
        headers["Authorization"] = f"Bearer {self.access_token()}"

        try:
            response = self.session.request(
                method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            raise DriveError(f"Drive request to {url} failed: {exc}") from exc

        if response.status_code == 401 and retry_auth:
            logger.info(
                "Drive returned 401; refreshing the access token and retrying once."
            )
            self._access_token = None
            self._expires_at = 0.0
            return self._call(method, url, retry_auth=False, **kwargs)

        body = _json_or_empty(response)
        if response.status_code >= 400:
            error = body.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise DriveError(
                f"Drive {method} {url.split('?')[0]} failed "
                f"(HTTP {response.status_code}): {message or response.text[:200]}"
            )
        return body

    # -- files -------------------------------------------------------------

    def find_by_name(self, name: str) -> str | None:
        """Returns the id of a non-trashed file with this name, if one exists.

        Drive permits duplicate names, so an upload that does not look for its
        predecessor accumulates a fresh copy of the report on every rebuild.
        """
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        clauses = [f"name = '{escaped}'", "trashed = false"]
        if self.folder_id:
            clauses.append(f"'{self.folder_id}' in parents")

        body = self._call(
            "GET",
            f"{API_BASE}/files",
            params={
                "q": " and ".join(clauses),
                "fields": "files(id,name)",
                "pageSize": 1,
                "spaces": "drive",
            },
        )
        files = body.get("files") or []
        return files[0].get("id") if files else None

    def upload(
        self, pdf_path: Path, name: str | None = None, share: bool = True
    ) -> DriveFile:
        """Uploads a PDF, replacing any earlier copy of the same name.

        Args:
            pdf_path: Local PDF to upload.
            name: Name to use in Drive. Defaults to the local file name.
            share: Grant link-readable access, which is the default because a
                report nobody can open has not been delivered.

        Returns:
            DriveFile: The uploaded file and its links.

        Raises:
            DriveError: If the upload or the sharing call fails.

        """
        path = Path(pdf_path)
        if not path.exists():
            raise DriveError(f"Cannot upload {path}: file does not exist.")

        target_name = name or path.name
        payload = path.read_bytes()
        existing_id = self.find_by_name(target_name)

        if existing_id:
            logger.info("Replacing Drive file %s (%s)", target_name, existing_id)
            body = self._call(
                "PATCH",
                f"{UPLOAD_BASE}/files/{existing_id}",
                params={"uploadType": "media", "fields": "id,name,webViewLink"},
                headers={"Content-Type": "application/pdf"},
                data=payload,
            )
        else:
            logger.info("Creating Drive file %s", target_name)
            metadata: dict[str, Any] = {
                "name": target_name,
                "mimeType": "application/pdf",
            }
            if self.folder_id:
                metadata["parents"] = [self.folder_id]
            boundary = f"grownxt-{uuid.uuid4().hex}"
            body = self._call(
                "POST",
                f"{UPLOAD_BASE}/files",
                params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
                headers={"Content-Type": f"multipart/related; boundary={boundary}"},
                data=_multipart_body(boundary, metadata, payload),
            )

        file_id = body.get("id")
        if not file_id:
            raise DriveError("Drive accepted the upload but returned no file id.")

        shared = self.share(file_id) if share else False
        return DriveFile(
            file_id=file_id,
            name=body.get("name") or target_name,
            view_link=body.get("webViewLink")
            or f"https://drive.google.com/file/d/{file_id}/view",
            preview_link=f"https://drive.google.com/file/d/{file_id}/preview",
            direct_link=f"https://drive.google.com/uc?export=download&id={file_id}",
            shared=shared,
            uploaded_at=_now_iso(),
        )

    def share(self, file_id: str) -> bool:
        """Makes a file readable by anyone holding the link.

        An existing grant is left alone: creating the same permission twice is
        an error on some Drive configurations, and re-granting establishes
        nothing that listing has not already established.

        Returns:
            bool: True when link-sharing is in place.

        """
        body = self._call(
            "GET",
            f"{API_BASE}/files/{file_id}/permissions",
            params={"fields": "permissions(id,type,role)"},
        )
        for permission in body.get("permissions") or []:
            if permission.get("type") == "anyone":
                return True

        self._call(
            "POST",
            f"{API_BASE}/files/{file_id}/permissions",
            params={"fields": "id"},
            json={"role": "reader", "type": "anyone"},
        )
        logger.info("Drive file %s shared as readable by link.", file_id)
        return True


def _multipart_body(boundary: str, metadata: dict[str, Any], payload: bytes) -> bytes:
    """Builds a ``multipart/related`` body: metadata part, then the PDF."""
    marker = f"--{boundary}".encode()
    return b"\r\n".join(
        [
            marker,
            b"Content-Type: application/json; charset=UTF-8",
            b"",
            json.dumps(metadata).encode("utf-8"),
            marker,
            b"Content-Type: application/pdf",
            b"",
            payload,
            f"--{boundary}--".encode(),
        ]
    )


def _json_or_empty(response: Any) -> dict[str, Any]:
    """Parses a JSON body, tolerating empty and non-JSON responses."""
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sha256(path: Path) -> str:
    """Returns the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(ticker: str) -> Path:
    """Returns the path recording what has been uploaded for one stock."""
    return stock_dir(ticker) / SIDECAR_NAME


def read_sidecar(ticker: str) -> dict[str, Any]:
    """Reads the upload record for one stock, or an empty mapping."""
    path = sidecar_path(ticker)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle) or {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable Drive sidecar %s: %s", path, exc)
        return {}


def _write_sidecar(ticker: str, record: dict[str, Any]) -> None:
    """Writes the upload record for one stock atomically."""
    path = sidecar_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    temporary.replace(path)


_STORE: DriveStore | None = None


def get_store() -> DriveStore:
    """Returns the process-wide store, so one access token serves every request."""
    global _STORE
    if _STORE is None:
        _STORE = DriveStore()
    return _STORE


def reset_store() -> None:
    """Drops the cached store, for tests and after a credential change."""
    global _STORE
    _STORE = None


def ensure_uploaded(
    pdf_path: Path,
    ticker: str,
    store: DriveStore | None = None,
    force: bool = False,
) -> DriveFile:
    """Returns the Drive copy of a report, uploading only what has changed.

    The PDF's content hash is the identity: a rebuild that produces identical
    bytes is not re-uploaded, and one that differs replaces the Drive file in
    place, so a link a user already holds keeps resolving to the current report.

    Args:
        pdf_path: Local PDF.
        ticker: Symbol the report belongs to.
        store: Drive store. Defaults to the process-wide one.
        force: Upload even when the hash is unchanged.

    Returns:
        DriveFile: The file in Drive.

    """
    path = Path(pdf_path)
    digest = _sha256(path)
    record = read_sidecar(ticker)

    if not force and record.get("pdf_sha256") == digest and record.get("file_id"):
        logger.debug("Drive copy of %s is current; skipping upload.", path.name)
        return DriveFile(
            file_id=record["file_id"],
            name=record.get("name") or path.name,
            view_link=record.get("view_link", ""),
            preview_link=record.get("preview_link", ""),
            direct_link=record.get("direct_link", ""),
            shared=bool(record.get("shared")),
            uploaded_at=record.get("uploaded_at", ""),
        )

    drive_file = (store or get_store()).upload(
        path, name=f"{safe_ticker(ticker)}_report.pdf"
    )
    payload = drive_file.as_dict()
    payload["pdf_sha256"] = digest
    _write_sidecar(ticker, payload)
    return drive_file
