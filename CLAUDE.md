# GrowNXT Server - Claude Developer Guide

Autonomous financial intelligence and institutional equity research platform. Combines live financial statement extraction (14 Vercel REST endpoints), Docling document layout parsing, Snowflake Arctic vector RAG, 20 deterministic accounting self-checks, and Typst PDF publication.

- **Live Hugging Face Space**: https://huggingface.co/spaces/StavanShah01/grownxt-server

---

## Architecture & Core Components

- **Dual-Track Reporting Engine (`reporting/engine.py`)**:
  - **Track 1 (Quantitative)**: Normalized income, balance sheet, and cash flow analysis (`reporting/snapshot.py`, `reporting/analytics.py`), composite scoring (`reporting/composites.py`), 20 deterministic self-checks (`reporting/selfcheck.py`), and SVG chart generation (`reporting/charts.py`).
  - **Track 2 (Qualitative RAG)**: Multi-document ingestion (`ingestion/`), semantic chunking with Docling layout parsing (`ingestion/chunker.py`), Qdrant in-memory vector index with Snowflake Arctic embeddings (`ingestion/indexer.py`), financial probes (`ingestion/rag/probes.py`), and SLM thematic synthesis (`ingestion/rag/synthesizer.py`).
  - **Publication**: Generates Typst markup (`reporting/typst_doc.py`) and compiles directly to PDF via the `typst` compiler.
- **Entry Points & Serving**:
  - `app.py`: Production entry point mounting Gradio UI + FastAPI ASGI + Flask WSGI (`a2wsgi`) on a unified server (default port `7860`).
  - `api/app.py`: Flask Application Factory (`create_app()`) exposing REST API routes and OpenAI-compatible `/v1/chat/completions` proxy with `<think>` token sanitization.
  - `scripts/generate_report.py`: Direct CLI batch runner for generating reports.
- **Storage & Distribution**:
  - `storage/gdrive.py`: Automated mirroring of generated PDFs to Google Drive.
  - `OUTPUT_DIR/<TICKER>/`: Unified per-stock workspace holding caches, charts, findings, and compiled PDFs.

---

## Repository Structure

```
GrowNXT_Server/
├── api/              # Flask application factory and search discovery
│   ├── app.py        # REST API endpoints & OpenAI reverse proxy
│   └── search.py     # External stock search and directory validation
├── core/             # Centralized paths and SLM configuration
│   ├── config.py     # System paths, safe_ticker folding, endpoint constants
│   └── llm_config.py # Hosted SLM integration and <think> token filtering
├── ingestion/        # Document fetching, chunking, and RAG pipeline
│   ├── catalog.py    # Document metadata catalog
│   ├── chunker.py    # Docling layout and Markdown chunking
│   ├── fetcher.py    # BSE/NSE document downloader
│   ├── indexer.py    # Qdrant vector store and Snowflake Arctic embeddings
│   └── rag/          # Probes, retriever, reranker, and synthesizer
├── reporting/        # Quantitative financial engine and Typst generation
│   ├── analytics.py  # DuPont 5-factor, solvency, liquidity, and CAGR
│   ├── charts.py     # SVG chart generation (matplotlib)
│   ├── composites.py # Altman Z-score, Piotroski F-score, and quality metrics
│   ├── engine.py     # Dual-track parallel report builder
│   ├── selfcheck.py  # 20 deterministic accounting balance checks
│   ├── snapshot.py   # Multi-period financial statement normalization
│   └── typst_doc.py  # Institutional Typst markup document builder
├── storage/          # Google Drive mirror and upload integration
├── scripts/          # CLI tools, pipeline verification, and checks
├── Dockerfile        # Container image definition (Python 3.12-slim + Caddy)
├── docker-compose.yml# Container orchestration
└── run.py / app.py   # Launchers
```

---

## Development Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install locked production dependencies
pip install -r requirements.lock

# Install development dependencies (ruff, pytest, mypy, pre-commit)
pip install -r requirements-dev.txt

# Run main platform (Gradio UI + REST API on port 7860)
python app.py

# Run standalone Flask API server (port 5000)
python api/app.py

# Generate report via CLI
python scripts/generate_report.py WIPRO
python scripts/generate_report.py WIPRO INFY --refresh --keep-build
```

---

## Testing & Verification

Run specialized verification suites in `scripts/` (built on `scripts/checks.py`):

```powershell
# Verify reporting engine, analytics, and self-checks
python scripts/verify_reporting.py

# Verify document download, caching, and storage contracts
python scripts/verify_documents.py

# Verify Docling layout parsing and chunking
python scripts/verify_chunker.py

# Verify ingestion and vector search indexing
python scripts/verify_ingestion.py

# Verify Google Drive authentication and upload flow
python scripts/verify_gdrive.py

# Run end-to-end multi-stock pipeline evaluation
python scripts/evaluate_full_pipeline.py
```

---

## Configuration & Environment Variables

Create a root `.env` file (refer to `.env.example`):

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `PORT` | Web server listening port | `7860` (or `5000` for Flask) |
| `GROWNXT_OUTPUT_DIR` | Base directory for per-ticker build artifacts | `./output` |
| `GROWNXT_LLM_API_URL` | Upstream OpenAI-compatible SLM endpoint | `https://llm.maqsoftware.net/v1` |
| `GROWNXT_LLM_API_KEY` | Bearer token for SLM endpoint (optional) | - |
| `GROWNXT_LLM_MODEL` | Active SLM model name | `qwen-3.8-27b` |
| `DEFAULT_CHAT_MODELS`| Comma-separated list of available models | `qwen-3.8-27b,gemma-4-31b` |
| `FINANCIAL_DATA_SERVICE_URL` | Live Vercel REST financial statement service | `https://financial-data-collector-qrxj.vercel.app` |
| `GDRIVE_CLIENT_ID` | Google Drive OAuth Client ID | - |
| `GDRIVE_CLIENT_SECRET` | Google Drive OAuth Client Secret | - |
| `GDRIVE_ROOT_FOLDER_ID`| Target folder ID in Google Drive | - |

---

## Code & API Conventions

- **Style**: MUST follow Google Python Style Guide with strict type annotations (`typing`) and docstrings.
- **Console & Encoding**: ALWAYS ensure UTF-8 output (`cli.console_utf8()`) to avoid Windows `cp1252` encoding crashes on currency symbols (`₹`) and typography.
- **REST Endpoints**:
  - `GET /api/search?q=<query>`: Fast in-process stock lookup.
  - `GET /api/stocks/<sym>/report`: Returns metadata and Google Drive link (compiles on demand).
  - `GET /api/stocks/<sym>/report/file`: Direct binary PDF stream (`?download=1` for attachment).
  - `POST /v1/chat/completions`: OpenAI-compatible streaming chat completion proxy.
- **Domain Errors**: Map domain exceptions to appropriate HTTP status codes via `STATUSES` in `api/app.py` (`ReportError` -> 404, `DriveAuthError` -> 503, `DriveError` -> 502).

---

## Zero-Hallucination & Reporting Rules

- **Deterministic Math**: Financial metrics, DuPont factors, solvency ratios, and growth rates MUST be calculated in Python (`reporting/analytics.py`), never generated freehand by LLMs.
- **Self-Check Enforcement**: `reporting/selfcheck.py` validates 20 accounting balance equations before report rendering. Failures MUST be logged.
- **Thinking Token Sanitization**: ALWAYS sanitize internal `<think>...</think>` tokens using `core.llm_config.clean_thinking_tokens` before emitting text to reports or streaming APIs.
- **Indian Denominations**: Financial figures MUST be formatted in standard Indian nomenclature (`₹ Cr`, `₹ Lakh`) via `reporting/fmt.py`.

---

## Docker & Hugging Face Spaces Deployment

- **Live Space**: [StavanShah01/grownxt-server](https://huggingface.co/spaces/StavanShah01/grownxt-server)
- **Container Architecture (`Dockerfile`)**:
  - Base: `python:3.12-slim` + system font/graphics libraries (`libgl1-mesa-glx`, `libglib2.0-0`, `caddy`) and CPU-optimized PyTorch.
  - User: Non-root user `user` (UID `1000`) required for Hugging Face Spaces runtime security.
  - Pre-cached Model: `Snowflake/snowflake-arctic-embed-m-v1.5` pre-downloaded during Docker build into image layer.
- **Hugging Face Spaces Lifecycle (`app.py`)**:
  - Listens on `0.0.0.0:7860` (standard Space port).
  - Multi-Framework Mount: Gradio 5 UI at `/` + Flask WSGI (`a2wsgi`) at `/v1` and `/api/llm` + FastAPI REST at `/api/*`.
  - ZeroGPU Compatibility: `@spaces.GPU` helper initialized to satisfy ZeroGPU supervisor when running on GPU hardware.
  - Stability Flags: `GRADIO_SSR_MODE=False` and `GRADIO_ANALYTICS_ENABLED=False` enforced to avoid SSR crashes.
- **Execution Modes (`entrypoint.sh`)**:
  - `APP_MODE=gradio` (default): Runs unified `python app.py` for Space and local dev.
  - `APP_MODE=caddy`: Runs Caddy proxy (port 7860) + Gunicorn (port 5000) for headless API serving.

---

## Common Gotchas & Non-obvious Constraints

- **Ticker Folding**: Indian tickers contain ampersands/hyphens (e.g. `M&M`). ALWAYS use `safe_ticker(sym)` from `core/config.py` when constructing directory paths.
- **Directory Lifecycle**: Build artifacts (`*.svg`, `*.typ`) are swept after compilation (`_sweep_build`), leaving only `<TICKER>_report.pdf` and `findings/` in `OUTPUT_DIR/<TICKER>/`.
- **No Database Engine**: Project is entirely file-backed and cached per-ticker in `OUTPUT_DIR/`. Do not introduce SQL or ORM dependencies.

---

## Definition of Done

1. **Implementation**: Validated against Google Python Style Guide and type checked.
2. **Verification**: Relevant verification suite in `scripts/verify_*.py` passes with zero failures.
3. **Compilation**: Typst compiles report without syntax errors (`python scripts/generate_report.py <TICKER>`).
4. **Security & Deployment**: No tokens or credentials committed; Docker container and Hugging Face Space (`app.py`) build and boot cleanly on port 7860.
5. **Contract**: API route compatibility and JSON schemas preserved.
