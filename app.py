"""GrowNXT Institutional Equity Research Platform -- Hugging Face Spaces & Local Entrypoint.

Runs the GrowNXT Flask API backend in a background worker and exposes an interactive
Gradio UI on port 7860 with multi-tab support:
  1. Institutional Report Generator (Typst PDF + Google Drive Delivery)
  2. Listed Stock Search
  3. Real-Time SLM Financial Analyst Chat (Direct Streaming via Qwen / Gemma)
  4. REST API Documentation & Reverse Proxy Status

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

# Completely disable experimental Node.js SSR in Gradio 5
os.environ["GRADIO_SSR_MODE"] = "False"

from dotenv import load_dotenv
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
from core.llm_config import (
    ACTIVE_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODELS,
    clean_thinking_tokens,
    stream_llm_response,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("grownxt.spaces")

INTERNAL_PORT = int(os.getenv("INTERNAL_PORT", "5000"))
HF_PORT = int(os.getenv("PORT", "7860"))

# ---------------------------------------------------------------------------
# 1. Background Flask Backend Daemon
# ---------------------------------------------------------------------------
flask_app = create_app()


def start_flask_worker() -> None:
    """Launches the internal Flask API server in a background thread."""
    logger.info("Starting internal GrowNXT Flask API backend on port %d...", INTERNAL_PORT)
    flask_app.run(
        host="127.0.0.1",
        port=INTERNAL_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# Start Flask as daemon thread before launching Gradio
flask_thread = threading.Thread(target=start_flask_worker, daemon=True, name="Flask-Worker")
flask_thread.start()
time.sleep(1.0)  # Brief grace period for socket binding


# ---------------------------------------------------------------------------
# 2. Gradio Business Logic & Handlers
# ---------------------------------------------------------------------------
def handle_stock_search(query: str) -> str:
    """Queries the internal search API for matching listed stocks."""
    if not query or len(query.strip()) < 2:
        return "⚠️ *Please enter at least 2 characters to search.*"

    try:
        url = f"http://127.0.0.1:{INTERNAL_PORT}/api/search?q={query.strip()}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return f"❌ Search error ({resp.status_code}): {resp.text}"

        data = resp.json()
        if not data:
            return f"No listed companies found matching **'{query}'**."

        lines = ["| Symbol | Company Name | Industry |", "| :--- | :--- | :--- |"]
        for item in data[:15]:
            sym = item.get("symbol", "-")
            name = item.get("name", "-")
            ind = item.get("industry", "N/A")
            lines.append(f"| **`{sym}`** | {name} | {ind} |")

        return "\n".join(lines)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return f"❌ Internal search error: {exc}"


def handle_generate_report(
    symbol: str,
    refresh: bool,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str, str]:
    """Generates the institutional equity report and returns the compiled PDF."""
    if not symbol or not symbol.strip():
        return None, "⚠️ *Please specify a stock ticker (e.g. INFY, WIPRO, TCS, M&M).*", ""

    sym = symbol.strip().upper()
    progress(0.1, desc=f"Initializing report pipeline for {sym}...")

    api_url = f"http://127.0.0.1:{INTERNAL_PORT}/api/stocks/{sym}/report"
    if refresh:
        api_url += "?refresh=1"

    progress(0.3, desc=f"Executing quantitative checks and RAG extraction for {sym}...")
    try:
        t0 = time.perf_counter()
        resp = requests.get(api_url, timeout=400)
        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            err_msg = f"❌ Report generation failed ({resp.status_code}):\n```json\n{resp.text}\n```"
            return None, err_msg, ""

        data = resp.json()
        drive_link = data.get("view_link") or data.get("drive_status") or "Drive not configured"
        was_built = data.get("generated", False)

        progress(0.85, desc=f"Fetching compiled PDF file bytes...")
        file_resp = requests.get(
            f"http://127.0.0.1:{INTERNAL_PORT}/api/stocks/{sym}/report/file",
            timeout=60,
        )
        if file_resp.status_code != 200:
            return None, f"❌ Failed to retrieve PDF bytes ({file_resp.status_code})", drive_link

        temp_pdf = os.path.join(os.path.expanduser("~"), f"{sym}_institutional_report.pdf")
        with open(temp_pdf, "wb") as f:
            f.write(file_resp.content)

        progress(1.0, desc="Report ready!")

        status_text = (
            f"✅ **Institutional Report Successfully Generated for `{sym}`!**\n\n"
            f"- **Execution Time**: `{elapsed:.2f}s` (Status: {'Compiled New' if was_built else 'Served from Cache'})\n"
            f"- **PDF File Size**: `{len(file_resp.content) / 1024:.1f} KB`\n"
            f"- **Google Drive Mirror**: {f'[{drive_link}]({drive_link})' if drive_link.startswith('http') else drive_link}"
        )

        drive_info = f"Google Drive: {drive_link}"
        return temp_pdf, status_text, drive_info
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

with gr.Blocks(title="GrowNXT Institutional Equity Platform", theme=gr.themes.Soft(), css=custom_css, ssr_mode=False) as demo:
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
# 4. Main Launch Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Launching GrowNXT Gradio Platform on 0.0.0.0:%d...", HF_PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=HF_PORT,
        show_api=False,
        share=False,
    )


