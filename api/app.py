"""Flask API for the GrowNXT platform: find a stock, get its report.

`/api/search` resolves a query to listed companies. `/api/stocks/<sym>/report`
returns the Google Drive link to that company's typeset PDF, compiling and
uploading it the first time it is asked for; `/report/file` serves the bytes
for clients that would rather not go through Drive.

Domain failures are raised, not branched on. The handlers registered in
`create_app` map each exception type to its status, which is what keeps every
route down to the happy path.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

from api.search import find
from core.config import OUTPUT_DIR, REQUIRED_ENV_VARS, report_path
from reporting.engine import ReportError, generate_report
from storage import gdrive

log = logging.getLogger(__name__)

ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
PDF_MIME = "application/pdf"
TRUTHY = ("1", "true", "yes")
NO_DRIVE = (
    "Google Drive is not configured. Run 'python -m scripts.gdrive_auth' "
    "after setting GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET."
)

# Domain failure -> status. Most specific first: DriveAuthError is a DriveError,
# and a lost grant needs re-consent (503) while a failed call may be retried.
STATUSES: Tuple[Tuple[type, int], ...] = (
    (ReportError, 404),
    (gdrive.DriveAuthError, 503),
    (gdrive.DriveError, 502),
)

Json = Tuple[Response, int]


def check_env() -> bool:
    """Reports whether every required environment variable is set."""
    load_dotenv()
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        log.error("missing environment variables: %s", ", ".join(missing))
        return False
    return True


def _flag(name: str) -> bool:
    """Reads a boolean query-string flag."""
    return request.args.get(name, "").strip().lower() in TRUTHY


def _sym(symbol: str) -> str:
    """Normalises a symbol taken from the URL."""
    return symbol.strip().upper()


def _file_url(sym: str) -> str:
    """The endpoint serving the raw PDF for a symbol."""
    return "/api/stocks/%s/report/file" % sym


def _pdf(sym: str, refresh: bool) -> Tuple[Path, bool]:
    """Returns the report PDF, and whether this call compiled it.

    Raises:
        ReportError: If the collector has nothing to report on, or Typst
            refuses the generated source.
    """
    pdf = report_path(sym)
    if pdf.exists() and not refresh:
        return pdf, False

    log.info("building report for %s (refresh=%s)", sym, refresh)
    return generate_report(sym, output_dir=OUTPUT_DIR, refresh=refresh), True


def _failed(exc: Exception) -> Json:
    """Renders a domain failure, with the status mapped from its type."""
    status = next(code for kind, code in STATUSES if isinstance(exc, kind))
    sym = _sym((request.view_args or {}).get("symbol", ""))
    log.warning("%s (%d) for %s: %s", type(exc).__name__, status, sym or "-", exc)

    body: Dict[str, Any] = {"success": False, "error": str(exc)}
    if sym:
        body["symbol"] = sym
        # The report itself may still be reachable even when Drive is not.
        body["pdf_endpoint"] = _file_url(sym)
    return jsonify(body), status


def create_app() -> Flask:
    """Builds the configured application."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {
        "origins": ORIGINS,
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }})

    @app.get("/api/search")
    def search() -> Json:
        """Companies matching `q`; a query under three characters matches none."""
        return jsonify(find(request.args.get("q", ""))), 200

    @app.get("/api/stocks/<symbol>/report")
    def report(symbol: str) -> Json:
        """The Drive link to this symbol's report.

        Compiled and uploaded on first request, then answered from the recorded
        links. `?refresh=1` rebuilds it and replaces the Drive file in place, so
        a link already shared keeps resolving to the current report.
        """
        sym, refresh = _sym(symbol), _flag("refresh")
        pdf, built = _pdf(sym, refresh)
        body = {
            "success": True,
            "symbol": sym,
            "generated": built,
            "pdf_endpoint": _file_url(sym),
        }

        if not gdrive.credentials_present():
            # Local runs without credentials still serve the PDF; say why there
            # is no link rather than returning a bare null.
            return jsonify(dict(body, drive=None, drive_status=NO_DRIVE)), 200

        up = gdrive.ensure_uploaded(pdf, sym, force=refresh)
        return jsonify(dict(body, drive=up.as_dict(),
                            view_link=up.view_link,
                            preview_link=up.preview_link)), 200

    @app.get("/api/stocks/<symbol>/report/file")
    def report_file(symbol: str) -> Response:
        """The PDF itself. `?download=1` sends it as an attachment."""
        pdf, _ = _pdf(_sym(symbol), _flag("refresh"))
        return send_file(pdf, mimetype=PDF_MIME, as_attachment=_flag("download"),
                         download_name=pdf.name, max_age=0)

    for kind, _ in STATUSES:
        app.register_error_handler(kind, _failed)

    @app.errorhandler(404)
    def unknown_route(_: Any) -> Json:
        return jsonify({"success": False, "error": "No such route"}), 404

    @app.errorhandler(500)
    def unexpected(_: Any) -> Json:
        return jsonify({"success": False, "error": "Internal server error"}), 500

    return app


app: Flask = create_app()


def main() -> None:
    """Server entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not check_env():
        log.error("startup aborted: environment incomplete")
        return

    port = int(os.getenv("PORT", "5000"))
    log.info("serving on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
