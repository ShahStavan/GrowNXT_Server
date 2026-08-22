"""GrowNXT Institutional Equity Research Platform -- Hugging Face Spaces & Local Entrypoint.

Combines FastAPI, Flask WSGI Middleware (a2wsgi), and Gradio 5 into a single production server:
  1. REST API Endpoints on /api/* and /v1/* (Flask via WSGIMiddleware mounted directly onto Gradio's FastAPI app)
  2. Native Gradio 5 UI on / (Report Generator, Stock Search, Live SLM Chat)
  3. Direct in-process execution for zero internal latency.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

# Completely disable experimental SSR and telemetry
os.environ["GRADIO_SSR_MODE"] = "False"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from dotenv import load_dotenv
from a2wsgi import WSGIMiddleware
import gradio as gr
import requests

# Patch gradio_client bug with Pydantic v2 boolean additionalProperties schemas
try:
    import gradio_client.utils as _client_utils

    _orig_schema_converter = _client_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _orig_schema_converter(schema, defs)
        except Exception:
            return "Any"

    _client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

from api.app import create_app
from api.search import find
from core.config import OUTPUT_DIR, report_path, safe_ticker
from core.llm_config import (
    ACTIVE_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODELS,
    clean_thinking_tokens,
    stream_llm_response,
)
from reporting.engine import generate_report
from storage import gdrive

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("grownxt.spaces")

HF_PORT = int(os.getenv("PORT", "7860"))

# ---------------------------------------------------------------------------
# 1. Initialize Flask Backend Application
# ---------------------------------------------------------------------------
flask_app = create_app()
wsgi_handler = WSGIMiddleware(flask_app)


# ---------------------------------------------------------------------------
# 2. Gradio Direct In-Process Handlers (Zero Latency)
# ---------------------------------------------------------------------------
def handle_stock_search(query: str) -> str:
    """Performs direct in-process search for listed stocks."""
    if not query or len(query.strip()) < 2:
        return "⚠️ *Please enter at least 2 characters to search.*"

    try:
        results = find(query.strip())
        if not results:
            return f"No listed companies found matching **'{query}'**."

        lines = ["| Symbol | Company Name | Industry |", "| :--- | :--- | :--- |"]
        for item in results[:15]:
            sym = item.get("symbol", "-")
            name = item.get("name", "-")
            ind = item.get("industry", "N/A")
            lines.append(f"| **`{sym}`** | {name} | {ind} |")

        return "\n".join(lines)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return f"❌ Search error: {exc}"


def handle_generate_report(
    symbol: str,
    refresh: bool,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str, str]:
    """Generates the institutional equity report in-process and returns the compiled PDF."""
    if not symbol or not symbol.strip():
        return None, "⚠️ *Please specify a stock ticker (e.g. INFY, WIPRO, TCS, M&M).*", ""

    sym = safe_ticker(symbol.strip().upper())
    progress(0.1, desc=f"Initializing report pipeline for {sym}...")

    try:
        t0 = time.perf_counter()
        progress(0.3, desc=f"Executing quantitative checks and RAG extraction for {sym}...")
        
        pdf_path = generate_report(sym, output_dir=OUTPUT_DIR, refresh=refresh)
        elapsed = time.perf_counter() - t0

        progress(0.85, desc="Checking Google Drive mirror...")
        drive_link = "Drive not configured"
        try:
            if gdrive.credentials_present():
                up = gdrive.ensure_uploaded(pdf_path, sym, force=refresh)
                drive_link = up.view_link or "Uploaded"
        except Exception as drive_exc:
            logger.warning("Drive upload error: %s", drive_exc)

        progress(1.0, desc="Report ready!")

        file_size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0

        status_text = (
            f"✅ **Institutional Report Successfully Generated for `{sym}`!**\n\n"
            f"- **Execution Time**: `{elapsed:.2f}s`\n"
            f"- **PDF File Size**: `{file_size_kb:.1f} KB`\n"
            f"- **Google Drive Mirror**: {f'[{drive_link}]({drive_link})' if drive_link.startswith('http') else drive_link}"
        )

        return str(pdf_path), status_text, drive_link
    except Exception as exc:
        logger.error("Report generation handler failed for %s: %s", sym, exc)
        return None, f"❌ Exception during generation: {exc}", ""


def handle_slm_chat(
    message: str,
    history: List[Dict[str, str]],
    model_choice: str,
) -> Generator[List[Dict[str, str]], None, None]:
    """Streams real-time token responses from the upstream SLM with thinking tokens removed."""
    if not message or not message.strip():
        yield history
        return

    selected_model = model_choice or ACTIVE_MODEL
    updated_history = list(history) if history else []
    updated_history.append({"role": "user", "content": message})
    updated_history.append({"role": "assistant", "content": ""})

    try:
        token_generator = stream_llm_response(
            prompt=message,
            model=selected_model,
            strip_thinking=True,
        )
        for token in token_generator:
            updated_history[-1]["content"] += token
            yield updated_history
    except Exception as exc:
        updated_history[-1]["content"] += f"\n\n❌ *SLM streaming error: {exc}*"
        yield updated_history


# ---------------------------------------------------------------------------
# 3. Gradio UI Layout Definition
# ---------------------------------------------------------------------------
available_models = [m.strip() for m in DEFAULT_CHAT_MODELS.split(",") if m.strip()]
if ACTIVE_MODEL not in available_models:
    available_models.insert(0, ACTIVE_MODEL)

custom_css = """
#main-title { text-align: center; margin-bottom: 0.5rem; }
.gr-button-primary { background-color: #0f766e !important; color: white !important; }
"""

with gr.Blocks(title="GrowNXT Institutional Equity Platform", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 📊 GrowNXT Institutional Equity Research & SLM Server", elem_id="main-title")
    gr.Markdown(
        "Institutional Fundamental Analysis · 20 Self-Checks · DuPont & Solvency Analytics · "
        "Multimodal Filing RAG · Typst PDF Publishing · Upstream SLM Proxy."
    )

    with gr.Tabs():
        # --- TAB 1: REPORT GENERATOR ---
        with gr.Tab("📄 Generate Equity Report"):
            with gr.Row():
                with gr.Column(scale=4):
                    ticker_in = gr.Textbox(
                        label="Stock Ticker Symbol",
                        placeholder="e.g. INFY, WIPRO, TCS, M&M, HDFCBANK",
                        value="INFY",
                        max_lines=1,
                    )
                    refresh_in = gr.Checkbox(
                        label="Force Ingestion Refresh (Re-download & re-parse filings)",
                        value=False,
                    )
                    gen_btn = gr.Button("🚀 Generate Institutional PDF Report", variant="primary")
                    gr.Markdown("""
                    > **Note**: Cold-run reports (new ticker) execute the full 14-endpoint financial normalization, 
                    > Docling layout parsing, Snowflake Arctic vector embeddings, and LLM thematic synthesis.
                    """)
                with gr.Column(scale=6):
                    status_out = gr.Markdown(label="Generation Status")
                    pdf_file_out = gr.File(label="Download Typeset PDF Report")
                    drive_out = gr.Textbox(label="Cloud Storage", interactive=False, visible=False)

            gen_btn.click(
                fn=handle_generate_report,
                inputs=[ticker_in, refresh_in],
                outputs=[pdf_file_out, status_out, drive_out],
                show_api=False,
            )

        # --- TAB 2: STOCK SEARCH ---
        with gr.Tab("🔍 Listed Stock Search"):
            with gr.Row():
                with gr.Column(scale=8):
                    search_in = gr.Textbox(
                        label="Company Name or Keyword",
                        placeholder="e.g. Infosys, Tata, Reliance, Mahindra, Bank",
                    )
                with gr.Column(scale=2):
                    search_btn = gr.Button("Search", variant="primary")

            search_results_out = gr.Markdown()
            search_btn.click(
                fn=handle_stock_search,
                inputs=[search_in],
                outputs=[search_results_out],
                show_api=False,
            )
            search_in.submit(
                fn=handle_stock_search,
                inputs=[search_in],
                outputs=[search_results_out],
                show_api=False,
            )

        # --- TAB 3: SLM CHAT (DIRECT STREAMING) ---
        with gr.Tab("🤖 SLM Financial Assistant"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Active SLM Model",
                    choices=available_models,
                    value=available_models[0] if available_models else "qwen-3.8-27b",
                    interactive=True,
                )
            chatbot = gr.Chatbot(label="Chat with SLM (Unbuffered Streaming)", height=450, type="messages")
            msg_in = gr.Textbox(
                label="Ask a financial research query...",
                placeholder="e.g. What are the key margin drivers and capex outlook for Indian IT services?",
            )
            with gr.Row():
                send_btn = gr.Button("Send Message", variant="primary")
                clear_btn = gr.ClearButton([msg_in, chatbot])

            send_btn.click(
                fn=handle_slm_chat,
                inputs=[msg_in, chatbot, model_dropdown],
                outputs=[chatbot],
                show_api=False,
            ).then(lambda: "", None, msg_in, show_api=False)

            msg_in.submit(
                fn=handle_slm_chat,
                inputs=[msg_in, chatbot, model_dropdown],
                outputs=[chatbot],
                show_api=False,
            ).then(lambda: "", None, msg_in, show_api=False)

        # --- TAB 4: API & PROXY INFO ---
        with gr.Tab("🔌 REST API & Reverse Proxy"):
            gr.Markdown(f"""
            ### Available REST Endpoints on this Deployment:
            * `GET /api/search?q=<query>` — Search listed stocks
            * `GET /api/stocks/<symbol>/report` — Generate/fetch report & Drive link
            * `GET /api/stocks/<symbol>/report/file` — Direct PDF file download
            * `POST /v1/chat/completions` — OpenAI-compatible SLM chat completions (Streaming supported)
            * `GET /v1/models` — List available SLM models

            ### Upstream Configuration:
            * **Active Model**: `{ACTIVE_MODEL}`
            * **Supported Chat Models**: `{DEFAULT_CHAT_MODELS}`
            * **Thinking Tokens Sanitization**: `Enabled (<think> stripped)`
            """)

# ---------------------------------------------------------------------------
# 4. Mount REST API Endpoints onto Gradio's FastAPI App before Launch
# ---------------------------------------------------------------------------
try:
    if hasattr(demo, "app") and demo.app is not None:
        demo.app.mount("/api", wsgi_handler)
        demo.app.mount("/v1", wsgi_handler)
except Exception as exc:
    logger.warning("Could not pre-mount on demo.app: %s", exc)

if __name__ == "__main__":
    logger.info("Launching GrowNXT Native Gradio Platform on 0.0.0.0:%d...", HF_PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=HF_PORT,
        show_api=False,
    )
