"""Flask API for the GrowNXT platform: find a stock, get its report.

`/api/search` resolves a query to listed companies. `/api/stocks/<sym>/report`
returns the Google Drive link to that company's typeset PDF, compiling and
uploading it the first time it is asked for; `/report/file` serves the bytes
for clients that would rather not go through Drive.

Domain failures are raised, not branched on. The handlers registered in
`create_app` map each exception type to its status, which is what keeps every
route down to the happy path.
"""

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from flask_cors import CORS

from api.search import find
from core.config import OUTPUT_DIR, REQUIRED_ENV_VARS, report_path
from core.llm_config import (
    ACTIVE_MODEL,
    API_URL,
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    REQUEST_TIMEOUT,
    _headers as llm_headers,
    clean_thinking_tokens,
)
from reporting.engine import ReportError, generate_report
from storage import gdrive

log = logging.getLogger(__name__)

ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
PDF_MIME = "application/pdf"
TRUTHY = ("1", "true", "yes")
NO_DRIVE = (
    "Google Drive is not configured. Run 'python -m scripts.gdrive_auth' "
    "after setting GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET."
)

# Domain failure -> status. Most specific first: DriveAuthError is a DriveError,
# and a lost grant needs re-consent (503) while a failed call may be retried.
STATUSES: tuple[tuple[type, int], ...] = (
    (ReportError, 404),
    (gdrive.DriveAuthError, 503),
    (gdrive.DriveError, 502),
)

Json = tuple[Response, int]


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
    return f"/api/stocks/{sym}/report/file"


def _pdf(sym: str, refresh: bool) -> tuple[Path, bool]:
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

    body: dict[str, Any] = {"success": False, "error": str(exc)}
    if sym:
        body["symbol"] = sym
        # The report itself may still be reachable even when Drive is not.
        body["pdf_endpoint"] = _file_url(sym)
    return jsonify(body), status


def create_app() -> Flask:
    """Builds the configured application."""
    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            },
            r"/v1/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            },
        },
    )

    @app.get("/api/search")
    @app.get("/search")
    def search() -> Json:
        """Companies matching `q`; a query under three characters matches none."""
        return jsonify(find(request.args.get("q", ""))), 200

    @app.get("/api/stocks/<symbol>/report")
    @app.get("/stocks/<symbol>/report")
    def report(symbol: str) -> Json:
        """The Drive link to this symbol's report."""
        sym, refresh = _sym(symbol), _flag("refresh")
        pdf, built = _pdf(sym, refresh)
        body = {
            "success": True,
            "symbol": sym,
            "generated": built,
            "pdf_endpoint": _file_url(sym),
        }

        if not gdrive.credentials_present():
            return jsonify(dict(body, drive=None, drive_status=NO_DRIVE)), 200

        up = gdrive.ensure_uploaded(pdf, sym, force=refresh)
        return jsonify(
            dict(
                body,
                drive=up.as_dict(),
                view_link=up.view_link,
                preview_link=up.preview_link,
            )
        ), 200

    @app.get("/api/stocks/<symbol>/report/file")
    @app.get("/stocks/<symbol>/report/file")
    def report_file(symbol: str) -> Response:
        """The PDF itself. `?download=1` sends it as an attachment."""
        pdf, _ = _pdf(_sym(symbol), _flag("refresh"))
        return send_file(
            pdf,
            mimetype=PDF_MIME,
            as_attachment=_flag("download"),
            download_name=pdf.name,
            max_age=0,
        )

    # --- OpenAI-Compatible Reverse Proxy Endpoints (Client / Frontend Safe) ---
    @app.get("/v1/models")
    @app.get("/api/llm/v1/models")
    @app.get("/models")
    def list_models() -> Json:
        """Lists supported SLM / LLM chat, embed, and rerank models."""
        model_names = [m.strip() for m in DEFAULT_CHAT_MODELS.split(",") if m.strip()]
        if DEFAULT_EMBED_MODEL:
            model_names.append(DEFAULT_EMBED_MODEL)
        if DEFAULT_RERANK_MODEL:
            model_names.append(DEFAULT_RERANK_MODEL)

        models_data = [
            {
                "id": m,
                "object": "model",
                "created": 1700000000,
                "owned_by": "grownxt-slm",
            }
            for m in model_names
        ]
        return jsonify({"object": "list", "data": models_data}), 200

    @app.post("/v1/chat/completions")
    @app.post("/api/llm/v1/chat/completions")
    @app.post("/chat/completions")
    def chat_completions() -> Response:
        """Secure reverse proxy for OpenAI-compatible chat completions with unbuffered streaming."""
        payload = request.get_json(force=True, silent=True) or {}
        if not payload.get("model"):
            payload["model"] = ACTIVE_MODEL

        is_streaming = bool(payload.get("stream", False))
        endpoint = f"{API_URL}/chat/completions"

        if is_streaming:

            def generate_stream() -> Generator[str, None, None]:
                try:
                    upstream_resp = requests.post(
                        endpoint,
                        json=payload,
                        headers=llm_headers(),
                        timeout=REQUEST_TIMEOUT,
                        stream=True,
                    )
                    upstream_resp.raise_for_status()

                    inside_thinking = False
                    buffer = ""

                    for line in upstream_resp.iter_lines():
                        if not line:
                            yield "\n"
                            continue
                        line_str = (
                            line.decode("utf-8") if isinstance(line, bytes) else line
                        )
                        if not line_str.startswith("data:"):
                            yield f"{line_str}\n"
                            continue
                        data_content = line_str[5:].strip()
                        if data_content == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break

                        try:
                            chunk = json.loads(data_content)
                            choices = chunk.get("choices") or []
                            if not choices:
                                yield f"{line_str}\n\n"
                                continue

                            delta = choices[0].get("delta") or {}
                            token = delta.get("content") or ""

                            # Stream token filtering: suppress internal thinking tags
                            buffer += token
                            if "<think>" in buffer:
                                inside_thinking = True
                                buffer = buffer.split("<think>", 1)[0]
                                if buffer:
                                    choices[0]["delta"]["content"] = buffer
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    buffer = ""

                            if inside_thinking:
                                if "</think>" in token:
                                    inside_thinking = False
                                    after_think = token.split("</think>", 1)[1]
                                    if after_think:
                                        choices[0]["delta"]["content"] = after_think
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                continue

                            choices[0]["delta"]["content"] = token
                            yield f"data: {json.dumps(chunk)}\n\n"
                            buffer = ""
                        except Exception:
                            yield f"{line_str}\n\n"

                except Exception as exc:
                    err_chunk = {
                        "error": {
                            "message": f"Proxy upstream error: {exc}",
                            "type": "proxy_error",
                            "code": 502,
                        }
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"

            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disables proxy buffering
            }
            return Response(stream_with_context(generate_stream()), headers=headers)

        # Non-streaming execution
        try:
            upstream_resp = requests.post(
                endpoint,
                json=payload,
                headers=llm_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            upstream_resp.raise_for_status()
            data = upstream_resp.json()

            # Sanitize thinking tokens from output
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if msg.get("content"):
                    msg["content"] = clean_thinking_tokens(str(msg["content"]))

            return jsonify(data), upstream_resp.status_code
        except requests.RequestException as exc:
            return jsonify(
                {
                    "error": {
                        "message": f"SLM upstream call failed: {exc}",
                        "type": "upstream_error",
                    }
                }
            ), 502
        except Exception as exc:
            return jsonify({"error": {"message": str(exc), "type": "proxy_error"}}), 500

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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    if not check_env():
        log.error("startup aborted: environment incomplete")
        return

    port = int(os.getenv("PORT", "5000"))
    log.info("serving on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
