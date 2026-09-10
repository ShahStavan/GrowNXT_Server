"""One-time Google Drive authorisation, and a health check for the stored grant.

Run once to mint a refresh token:

    venv/Scripts/python.exe -m scripts.gdrive_auth

The browser opens Google's consent screen, the code comes back to a loopback
server on this machine, and the refresh token is written to ``.gdrive_token.json``
(git-ignored). The server picks it up from there on its next request -- nothing
has to be pasted into an environment variable.

Check an existing grant without touching the consent screen:

    venv/Scripts/python.exe -m scripts.gdrive_auth --check

Why this exists at all: access tokens are refreshed silently by
``storage/gdrive.py``, but a *refresh* token can only be minted with a human at
the consent screen. Google issues refresh tokens with a 7-day life while the
OAuth app is in Testing status, so publish the consent screen to Production and
this command becomes a once-ever step rather than a weekly chore.
"""

import argparse
import contextlib
import http.server
import logging
import os
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import cli  # noqa: E402
from storage.gdrive import (  # noqa: E402
    SCOPE,
    DriveAuthError,
    DriveError,
    DriveStore,
    _stored_token,
    _token_file,
    save_refresh_token,
)

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

logger = logging.getLogger("gdrive_auth")

SETUP_STEPS = """Set GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET first.

  1. https://console.cloud.google.com/ -> new project
  2. APIs & Services -> Library -> enable 'Google Drive API'
  3. OAuth consent screen -> External -> add yourself as a user, then
     PUBLISH it (an app left in Testing expires refresh tokens after 7 days)
  4. Credentials -> Create OAuth client ID -> Desktop app
  5. Put the id and secret in .env as GDRIVE_CLIENT_ID /
     GDRIVE_CLIENT_SECRET, then run this again
"""


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the single redirect Google makes back to this machine."""

    code = None
    error = None

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _CallbackHandler.code = (params.get("code") or [None])[0]
        _CallbackHandler.error = (params.get("error") or [None])[0]

        body = (
            b"<html><body style='font-family:sans-serif;padding:2rem'>"
            b"<h2>GrowNXT &mdash; Google Drive authorised</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
            if _CallbackHandler.code
            else b"<html><body><h2>Authorisation failed</h2></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Silences the default stderr access log."""


def _free_port() -> int:
    """Returns a port the loopback callback server can bind."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _post_form(url: str, fields: dict) -> dict:
    """POSTs a form and returns the parsed JSON body."""
    import json

    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Token exchange failed (HTTP {exc.code}): {detail}") from exc


def authorise(client_id: str, client_secret: str) -> str:
    """Runs the consent flow and returns a refresh token.

    Args:
        client_id: OAuth client id of a Desktop-type client.
        client_secret: Matching client secret.

    Returns:
        str: The refresh token.
    """
    port = _free_port()
    redirect_uri = f"http://127.0.0.1:{port}/"

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        # Without these two Google returns no refresh token on a repeat consent.
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.timeout = 5

    def serve_until_answered(deadline: float) -> None:
        """Serves requests until the redirect arrives, or time runs out.

        Handling a single request is not enough: a browser may prefetch
        `/favicon.ico` against the same port, which would consume the one
        request and leave the authorisation code unread.
        """
        while time.time() < deadline:
            if _CallbackHandler.code or _CallbackHandler.error:
                return
            server.handle_request()

    _CallbackHandler.code = None
    _CallbackHandler.error = None
    deadline = time.time() + 300
    thread = threading.Thread(
        target=serve_until_answered, args=(deadline,), daemon=True
    )
    thread.start()

    print(f"Opening Google's consent screen. If nothing opens, visit:\n\n{auth_url}\n")
    with contextlib.suppress(Exception):  # noqa: BLE001 - a headless box just uses the printed URL
        webbrowser.open(auth_url)

    thread.join(timeout=305)
    server.server_close()

    if _CallbackHandler.error:
        raise SystemExit(f"Google returned an error: {_CallbackHandler.error}")
    if not _CallbackHandler.code:
        raise SystemExit("No authorisation code received within 5 minutes.")

    payload = _post_form(
        TOKEN_ENDPOINT,
        {
            "code": _CallbackHandler.code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        raise SystemExit(
            "Google returned no refresh token. Revoke this app's access at "
            "https://myaccount.google.com/permissions and run this again."
        )
    return refresh_token


def _report_missing() -> None:
    """Names which credential is absent, without printing any of their values."""
    have_id = bool(os.getenv("GDRIVE_CLIENT_ID", "").strip())
    have_secret = bool(os.getenv("GDRIVE_CLIENT_SECRET", "").strip())
    stored = _stored_token()
    have_token = bool(stored or os.getenv("GDRIVE_REFRESH_TOKEN", "").strip())

    for label, present in (
        ("GDRIVE_CLIENT_ID", have_id),
        ("GDRIVE_CLIENT_SECRET", have_secret),
        ("refresh token", have_token),
    ):
        print(f"  [{'ok ' if present else '   '}] {label}")

    if have_id and have_secret and not have_token:
        print(
            "\nThe client credentials are in place; what is missing is the "
            "refresh token,\nwhich only the consent screen can mint. Run:\n\n"
            "    venv/Scripts/python.exe -m scripts.gdrive_auth\n"
        )


def check() -> int:
    """Verifies the stored grant still mints access tokens.

    Returns:
        int: Process exit code.
    """
    try:
        store = DriveStore()
    except DriveAuthError:
        print("Google Drive is not fully configured:\n")
        _report_missing()
        return 1

    try:
        store.access_token()
    except DriveAuthError as exc:
        print(f"GRANT REJECTED: {exc}")
        return 1
    except DriveError as exc:
        print(f"could not reach Google: {exc}")
        return 1

    print(f"ok: the stored refresh token still works (token file: {_token_file()})")
    if store.folder_id:
        print(f"uploads target folder {store.folder_id}")
    else:
        print("uploads land in the account root (set GDRIVE_FOLDER_ID to change that)")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    p = argparse.ArgumentParser(description="Authorise GrowNXT against Google Drive.")
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify the stored refresh token instead of re-authorising.",
    )
    p.add_argument("--client-id", help="Overrides GDRIVE_CLIENT_ID.")
    p.add_argument("--client-secret", help="Overrides GDRIVE_CLIENT_SECRET.")
    return p


def main() -> int:
    """Runs the consent flow, or checks the grant already stored."""
    args = _parser().parse_args()
    cli.setup(logging.INFO, cli.PLAIN)
    load_dotenv()

    if args.check:
        return check()

    client_id = args.client_id or os.getenv("GDRIVE_CLIENT_ID", "").strip()
    secret = args.client_secret or os.getenv("GDRIVE_CLIENT_SECRET", "").strip()
    if not (client_id and secret):
        print(SETUP_STEPS, file=sys.stderr)
        return 1

    path = save_refresh_token(authorise(client_id, secret))
    print(f"\nRefresh token saved to {path}")
    print("The server reads it from there; no restart needed for the next request.")
    return check()


if __name__ == "__main__":
    sys.exit(main())
