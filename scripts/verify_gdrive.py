"""Verification for the Google Drive delivery path.

Runs against an in-memory stand-in, so upload, sharing, token refresh and
caching are verified with no credentials. The invariants are those that fail
quietly: a second copy of every report accumulating, one uploaded but never
shared, a link dead after a rebuild, a revoked grant retried forever.
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.config as config  # noqa: E402
import storage.gdrive as gdrive  # noqa: E402
from scripts import cli  # noqa: E402
from scripts.checks import Report, banner  # noqa: E402
from storage.gdrive import (  # noqa: E402
    API_BASE,
    TOKEN_URL,
    UPLOAD_BASE,
    DriveAuthError,
    DriveError,
    DriveStore,
    ensure_uploaded,
)

PDF_BYTES = b"%PDF-1.7\n" + b"x" * 512 + b"\n%%EOF\n"


class FakeResponse:
    """Minimal stand-in for a ``requests`` response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        """Returns the decoded JSON body, raising when there is none."""
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeDrive:
    """An in-memory Drive: files with names, bytes, parents and permissions."""

    def __init__(self, token_responses=None):
        self.files = {}
        self.permissions = {}
        self.calls = []
        self.next_id = 1
        self.token_responses = list(token_responses or [])
        self.token_calls = 0

    # -- transport ---------------------------------------------------------

    def post(self, url, data=None, json=None, timeout=None, headers=None):  # noqa: ARG002 - mirrors the real transport's signature
        """Handles the token endpoint; everything else goes through request()."""
        if url == TOKEN_URL:
            self.token_calls += 1
            if self.token_responses:
                return self.token_responses.pop(0)
            return FakeResponse(
                200, {"access_token": f"at-{self.token_calls}", "expires_in": 3600}
            )
        return self.request("POST", url, data=data, json=json, headers=headers)

    def request(
        self,
        method,
        url,
        headers=None,
        params=None,
        data=None,
        json=None,
        timeout=None,  # noqa: ARG002 - mirrors the real transport's signature
    ):
        """Records the call and returns the queued fake response."""
        params = params or {}
        self.calls.append((method, url, params, headers or {}))

        if method == "GET" and url == f"{API_BASE}/files":
            return self._list(params.get("q", ""))
        if method == "POST" and url == f"{UPLOAD_BASE}/files":
            return self._create(data)
        if method == "PATCH" and url.startswith(f"{UPLOAD_BASE}/files/"):
            return self._update(url.rsplit("/", 1)[-1], data)
        if url.endswith("/permissions"):
            file_id = url.rsplit("/", 2)[-2]
            if method == "GET":
                return FakeResponse(
                    200, {"permissions": self.permissions.get(file_id, [])}
                )
            if method == "POST":
                grant = dict(json or {})
                grant["id"] = f"perm-{len(self.permissions.get(file_id, [])) + 1}"
                self.permissions.setdefault(file_id, []).append(grant)
                return FakeResponse(200, {"id": grant["id"]})
        return FakeResponse(404, {"error": {"message": f"unrouted {method} {url}"}})

    # -- behaviour ---------------------------------------------------------

    def _list(self, query):
        name = None
        if "name = '" in query:
            name = query.split("name = '", 1)[1].split("'", 1)[0]
        parent = None
        if " in parents" in query:
            parent = (
                query.split("'", 1)[1].split("'", 1)[0]
                if query.startswith("'")
                else None
            )
            for part in query.split(" and "):
                if part.endswith(" in parents"):
                    parent = part.split("'")[1]
        matches = [
            {"id": fid, "name": meta["name"]}
            for fid, meta in self.files.items()
            if meta["name"] == name
            and (parent is None or parent in meta.get("parents", []))
        ]
        return FakeResponse(200, {"files": matches})

    def _create(self, body):
        metadata = {}
        if body:
            text = body.decode("utf-8", "replace")
            start = text.find("{")
            end = text.find("}", start)
            if start >= 0 and end > start:
                metadata = json.loads(text[start : end + 1])
        file_id = f"file-{self.next_id}"
        self.next_id += 1
        self.files[file_id] = {
            "name": metadata.get("name", "unnamed"),
            "parents": metadata.get("parents", []),
            "bytes": body or b"",
        }
        return FakeResponse(
            200,
            {
                "id": file_id,
                "name": self.files[file_id]["name"],
                "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
            },
        )

    def _update(self, file_id, body):
        if file_id not in self.files:
            return FakeResponse(404, {"error": {"message": "no such file"}})
        self.files[file_id]["bytes"] = body or b""
        self.files[file_id]["updates"] = self.files[file_id].get("updates", 0) + 1
        return FakeResponse(
            200,
            {
                "id": file_id,
                "name": self.files[file_id]["name"],
                "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
            },
        )


def _store(fake, **kwargs):
    """Builds a store wired to a fake Drive."""
    options = {
        "client_id": "cid",
        "client_secret": "secret",
        "refresh_token": "rt-original",
        "folder_id": "",
        "session": fake,
    }
    options.update(kwargs)
    return DriveStore(**options)


def _write_pdf(directory: Path, body: bytes = PDF_BYTES) -> Path:
    """Writes a PDF into a directory and returns its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "WIPRO_report.pdf"
    path.write_bytes(body)
    return path


def check_configuration(r: Report) -> None:
    """Configuration and credential handling."""
    saved = {
        k: os.environ.pop(k, None)
        for k in (
            "GDRIVE_CLIENT_ID",
            "GDRIVE_CLIENT_SECRET",
            "GDRIVE_REFRESH_TOKEN",
            "GDRIVE_TOKEN_FILE",
        )
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GDRIVE_TOKEN_FILE"] = str(Path(tmp) / "token.json")
            r.check(
                "config.not_configured_reports_false",
                not gdrive.credentials_present(),
                "no credentials present",
            )
            r.raises(
                "config.missing_credentials_raise_auth_error",
                DriveAuthError,
                lambda: DriveStore(client_id="", client_secret="", refresh_token=""),
            )

            os.environ["GDRIVE_CLIENT_ID"] = "cid"
            os.environ["GDRIVE_CLIENT_SECRET"] = "secret"
            os.environ["GDRIVE_REFRESH_TOKEN"] = "rt"
            r.check(
                "config.env_credentials_detected",
                gdrive.credentials_present(),
                "client id, secret and refresh token seen",
            )

            os.environ.pop("GDRIVE_REFRESH_TOKEN")
            path = gdrive.save_refresh_token("rt-from-file")
            r.check(
                "config.token_file_is_read_back",
                gdrive.credentials_present()
                and gdrive._stored_token() == "rt-from-file",
                f"token file {Path(path).name} supplies the grant",
            )

            store = _store(FakeDrive(), refresh_token=None)
            r.check(
                "config.token_file_beats_env",
                store.refresh_token == "rt-from-file",
                "the persisted token is preferred over the environment",
            )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_tokens(r: Report) -> None:
    """Access-token lifecycle: the part that keeps a long-running server alive."""
    fake = FakeDrive()
    store = _store(fake)
    first = store.access_token()
    second = store.access_token()
    r.check(
        "token.access_token_is_cached",
        first == second and fake.token_calls == 1,
        f"two calls, one refresh ({first})",
    )

    store._expires_at = time.time() - 1
    third = store.access_token()
    r.check(
        "token.expiry_triggers_refresh",
        third != first and fake.token_calls == 2,
        "expired token replaced without intervention",
    )

    fake = FakeDrive(
        token_responses=[
            FakeResponse(
                400,
                {
                    "error": "invalid_grant",
                    "error_description": "Token has been expired or revoked.",
                },
            ),
        ]
    )
    store = _store(fake)
    r.raises(
        "token.invalid_grant_is_terminal",
        DriveAuthError,
        store.access_token,
        "revoked grant surfaces as DriveAuthError",
    )

    # A fresh store, because the rejection above consumed the queued response.
    store = _store(
        FakeDrive(
            token_responses=[
                FakeResponse(
                    400,
                    {
                        "error": "invalid_grant",
                        "error_description": "Token has been expired or revoked.",
                    },
                ),
            ]
        )
    )
    message = ""
    try:
        store.access_token()
    except DriveAuthError as exc:
        message = str(exc)
    r.check(
        "token.invalid_grant_says_how_to_fix",
        "scripts.gdrive_auth" in message and "Testing status" in message,
        "names the re-auth command and the 7-day cause",
    )

    fake = FakeDrive(token_responses=[FakeResponse(503, {}, text="backend error")])
    store = _store(fake)
    r.raises(
        "token.transient_failure_is_not_auth_error",
        DriveError,
        store.access_token,
        "a 503 is retryable, not a lost grant",
    )
    fake = FakeDrive(token_responses=[FakeResponse(503, {}, text="backend error")])
    store = _store(fake)
    try:
        store.access_token()
    except DriveAuthError:
        r.check(
            "token.transient_failure_not_misreported",
            False,
            "503 raised DriveAuthError",
        )
    except DriveError:
        r.check(
            "token.transient_failure_not_misreported",
            True,
            "DriveError, so the caller may retry",
        )

    saved = os.environ.get("GDRIVE_TOKEN_FILE")
    with tempfile.TemporaryDirectory() as tmp:
        token_file = Path(tmp) / "token.json"
        os.environ["GDRIVE_TOKEN_FILE"] = str(token_file)
        try:
            fake = FakeDrive(
                token_responses=[
                    FakeResponse(
                        200,
                        {
                            "access_token": "at-rotated",
                            "expires_in": 3600,
                            "refresh_token": "rt-rotated",
                        },
                    )
                ]
            )
            store = _store(fake)
            store.access_token()
            persisted = json.loads(token_file.read_text(encoding="utf-8")).get(
                "refresh_token"
            )
            r.check(
                "token.rotated_refresh_token_persisted",
                store.refresh_token == "rt-rotated" and persisted == "rt-rotated",
                "a rotated grant survives a restart",
            )
        finally:
            if saved is None:
                os.environ.pop("GDRIVE_TOKEN_FILE", None)
            else:
                os.environ["GDRIVE_TOKEN_FILE"] = saved

    # A 401 mid-session: one silent retry, no error surfaced to the caller.
    fake = FakeDrive()
    calls = {"n": 0}
    original = fake.request

    def flaky(method, url, **kwargs):
        """Fails the first call with 401, then succeeds."""
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(401, {"error": {"message": "Invalid Credentials"}})
        return original(method, url, **kwargs)

    fake.request = flaky
    store = _store(fake)
    store.find_by_name("WIPRO_report.pdf")
    r.check(
        "token.401_retried_once_with_a_fresh_token",
        calls["n"] == 2 and fake.token_calls == 2,
        "one retry after re-minting the access token",
    )


def check_upload(r: Report) -> None:
    """Upload, replacement and sharing."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = _write_pdf(Path(tmp))

        fake = FakeDrive()
        store = _store(fake)
        first = store.upload(pdf)
        r.check(
            "upload.creates_when_absent",
            len(fake.files) == 1 and first.file_id in fake.files,
            f"one file created ({first.file_id})",
        )
        body = fake.files[first.file_id]["bytes"]
        r.check(
            "upload.multipart_carries_metadata_and_pdf",
            b"WIPRO_report.pdf" in body and b"%PDF-1.7" in body,
            f"{len(body)} bytes, metadata part and PDF part present",
        )

        second = store.upload(pdf)
        r.check(
            "upload.replaces_rather_than_duplicating",
            len(fake.files) == 1 and second.file_id == first.file_id,
            f"second upload reused {second.file_id}",
        )
        r.check(
            "upload.replacement_is_a_media_patch",
            fake.files[first.file_id].get("updates") == 1,
            "existing file updated in place, so the link still resolves",
        )

        r.check(
            "share.link_sharing_is_the_default",
            any(
                p.get("type") == "anyone" and p.get("role") == "reader"
                for p in fake.permissions.get(first.file_id, [])
            ),
            "anyone-with-link reader granted on upload",
        )
        r.check(
            "share.existing_grant_is_not_recreated",
            len(fake.permissions.get(first.file_id, [])) == 1,
            "one permission after two uploads",
        )

        unshared = _store(FakeDrive())
        result = unshared.upload(pdf, share=False)
        r.check(
            "share.can_be_declined_explicitly",
            result.shared is False,
            "share=False leaves the file private",
        )

        r.check(
            "links.preview_link_is_embeddable",
            first.preview_link
            == f"https://drive.google.com/file/d/{first.file_id}/preview",
            first.preview_link,
        )
        r.check(
            "links.direct_link_points_at_the_file",
            first.file_id in first.direct_link
            and "export=download" in first.direct_link,
            first.direct_link,
        )

        fake = FakeDrive()
        store = _store(fake, folder_id="folder-123")
        placed = store.upload(pdf)
        r.check(
            "upload.folder_id_is_applied",
            fake.files[placed.file_id]["parents"] == ["folder-123"],
            "file parented to the configured folder",
        )

        fake = FakeDrive()
        store = _store(fake)
        store.find_by_name("O'Brien & Sons_report.pdf")
        query = [c for c in fake.calls if c[0] == "GET"][0][2].get("q", "")
        r.check(
            "upload.name_query_escapes_quotes",
            "O\\'Brien" in query,
            "an apostrophe cannot break the query",
        )

        missing = _store(FakeDrive())
        r.raises(
            "upload.missing_file_is_refused",
            DriveError,
            lambda: missing.upload(Path(tmp) / "nope.pdf"),
        )


def check_caching(r: Report) -> None:
    """The sidecar: what stops every request re-uploading the same report."""
    original_output = config.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.OUTPUT_DIR = Path(tmp)
        try:
            ticker = "WIPRO"
            pdf = _write_pdf(config.stock_dir(ticker))

            fake = FakeDrive()
            store = _store(fake)
            first = ensure_uploaded(pdf, ticker, store=store)
            sidecar = gdrive.sidecar_path(ticker)
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            r.check(
                "cache.sidecar_written",
                sidecar.exists() and record.get("file_id") == first.file_id,
                f"{sidecar.name} records the file id and links",
            )
            r.check(
                "cache.sidecar_records_content_hash",
                len(record.get("pdf_sha256", "")) == 64,
                "identity is the PDF's own hash",
            )

            uploads_before = len(fake.calls)
            again = ensure_uploaded(pdf, ticker, store=store)
            r.check(
                "cache.unchanged_pdf_is_not_reuploaded",
                len(fake.calls) == uploads_before and again.file_id == first.file_id,
                "no Drive calls for an unchanged report",
            )

            pdf.write_bytes(PDF_BYTES + b"revised\n")
            changed = ensure_uploaded(pdf, ticker, store=store)
            r.check(
                "cache.changed_pdf_is_reuploaded",
                changed.file_id == first.file_id
                and fake.files[first.file_id].get("updates") == 1,
                "same file id, new bytes -- the shared link is stable",
            )
            new_record = json.loads(sidecar.read_text(encoding="utf-8"))
            r.check(
                "cache.hash_advances_with_the_rebuild",
                new_record["pdf_sha256"] != record["pdf_sha256"],
                "sidecar tracks the new content",
            )

            forced_before = fake.files[first.file_id].get("updates")
            ensure_uploaded(pdf, ticker, store=store, force=True)
            r.check(
                "cache.force_reuploads_identical_bytes",
                fake.files[first.file_id].get("updates") == forced_before + 1,
                "force=True bypasses the hash check",
            )

            gdrive.sidecar_path(ticker).write_text("{ not json", encoding="utf-8")
            recovered = ensure_uploaded(pdf, ticker, store=store)
            r.check(
                "cache.corrupt_sidecar_is_survivable",
                bool(recovered.file_id),
                "an unreadable sidecar re-uploads instead of crashing",
            )
        finally:
            config.OUTPUT_DIR = original_output


def main() -> int:
    """Runs every check and returns a process exit code."""
    cli.setup(logging.CRITICAL)
    banner("Google Drive delivery verification (offline)")

    report = Report()
    for title, group in (
        ("Configuration", check_configuration),
        ("Token handling", check_tokens),
        ("Upload and sharing", check_upload),
        ("Upload caching", check_caching),
    ):
        report.section(title)
        group(report)

    print(report.render())
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
