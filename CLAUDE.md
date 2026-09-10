# GrowNXT Server - Claude Developer Guide

Autonomous financial intelligence and institutional equity research platform. Combines live financial statement extraction (14 Vercel REST endpoints), Docling document layout parsing, 20 deterministic accounting self-checks, and Typst PDF publication.

- **Live Hugging Face Space**: https://huggingface.co/spaces/StavanShah01/grownxt-server

---

## ⚠️ Status: the vector pipeline was removed on 2026-09-10

The qualitative half of this platform is **mid-rebuild**. The Snowflake Arctic / Qdrant vector
pipeline, the Nifty 50 batch orchestrator and the LangGraph RAG pipeline were deleted ahead of a
vectorless replacement. **Large parts of this file below still describe that removed code.**

- **Design**: `.claude/specs/vectorless-qualitative-rag.md`
- **Implementation plan**: `.claude/plans/vectorless-qualitative-rag.md`
- **Recover any deleted file from**: the commit before `refactor(ingestion)!: remove the vector
  pipeline`, on branch `feat/vectorless-qualitative-rag`.

**Gone**: `ingestion/{batch,chunker,stages,indexer,nifty50,notify,runlog}.py`,
`ingestion/graph/`, `ingestion/rag/`, `api/embeddings.py`, `scripts/{embed_nifty50,
verify_nifty50_embeddings,verify_chunker,evaluate_full_pipeline,ingest_documents,
verify_ingestion}.py`. With them: the `/api/embeddings/nifty50/*` routes, every `QDRANT_*` and
`GROWNXT_EMBED_*` variable, and the `python -m scripts.embed_nifty50` command family.

**Survives, and is the base for the rebuild**: `ingestion/catalog.py`, `ingestion/fetcher.py`,
`ingestion/documents/` (download, Docling extraction, storage, page sections), all of
`reporting/`, `core/`, `api/app.py`, `storage/`.

**Track 2 is commented out**, not deleted, in `reporting/engine.py` — the call shape is on
record for the replacement, which restores the same seam and the same `findings.json` schema. A
`findings.json` already on disk still renders; nothing regenerates one today.

**Current gate**: `python scripts/parity_gate.py` — 12 of 14, with two pre-existing failures
(`import app` on a Gradio 6 / Gradio 5 mismatch, and 2 of 51 Docling heading checks in
`verify_documents.py`). See the plan's F11.

---

## Architecture & Core Components

- **Dual-Track Reporting Engine (`reporting/engine.py`)**:
  - **Track 1 (Quantitative)**: Normalized income, balance sheet, and cash flow analysis (`reporting/snapshot.py`, `reporting/analytics.py`), composite scoring (`reporting/composites.py`), 20 deterministic self-checks (`reporting/selfcheck.py`), and SVG chart generation (`reporting/charts.py`).
  - **Track 2 (Qualitative RAG)**: Multi-document ingestion (`ingestion/`), semantic chunking with Docling layout parsing (`ingestion/chunker.py`), Qdrant in-memory vector index with Snowflake Arctic embeddings (`ingestion/indexer.py`), financial probes (`ingestion/rag/probes.py`), and SLM thematic synthesis (`ingestion/rag/synthesizer.py`).
  - **Publication**: Generates Typst markup (`reporting/typst_doc.py`) and compiles directly to PDF via the `typst` compiler.
- **Nifty 50 Batch Embedding (`ingestion/batch.py`, `scripts/embed_nifty50.py`)**:
  - Embeds each constituent's newest annual report, concall transcript and investor presentation into **its own Qdrant collection** (`<QDRANT_COLLECTION_NAME>_<ticker>`) and mirrors them to `output/<TICKER>/vectors.npz` + `payloads.json` (the retriever's Tier-1 fast path).
  - Two-layer change detection: Layer 1 diffs the catalogue's period-derived `doc_id`s against `output/<TICKER>/state.json` (new / failed / chunker-or-extractor version bump / collection change); Layer 2 is the indexer's fingerprint check. A steady-state run is 50 catalogue requests and nothing else.
  - Constituents come from `ticker_mapping.csv` via `ingestion/nifty50.py`; run state lives in `output/_nifty50/run_manifest.json`; logs in `logs/nifty50/` (`ingestion/runlog.py`: rolling log, per-run `run.log`, `events.jsonl`, `summary.json`).
  - Runs detached (`--background`) or via `POST /api/embeddings/nifty50/run`. Notifies once on completion, and once per constituent as it finishes (`ingestion/notify.py`), both on the same webhook.
- **Compute Resolution (`core/hardware.py`)**:
  - One profile per run — device, extraction threads, embed batch size, embed precision — detected from the machine and applied to every stage: Docling's `AcceleratorOptions`, torch's intra-op pool, and the `SentenceTransformer`. `profile()` resolves, `configure()` applies; both are safe without torch installed.
  - Exists because each library defaults badly on its own: Docling pins **4 threads on any machine** and `device="auto"`, which resolves to CPU whenever torch came from the CPU wheel index — so a GPU can sit idle for a fifty-hour run with nothing said. `--hardware` reports the resolution up front, and `idle_gpu` names that exact case.
  - The resolved profile is recorded in `RunManifest.hardware` and in the `run_started` event.
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
├── core/             # Centralized paths, compute, and SLM configuration
│   ├── config.py     # System paths, safe_ticker folding, endpoint constants
│   ├── hardware.py   # Device/threads/batch resolution for Docling, torch, the embedder
│   └── llm_config.py # Hosted SLM integration and <think> token filtering
├── ingestion/        # Document fetching, chunking, and RAG pipeline
│   ├── batch.py      # Nifty 50 batch orchestration: run_batch(), manifest, detached launch
│   ├── catalog.py    # Document metadata catalog
│   ├── chunker.py    # Docling layout and Markdown chunking
│   ├── documents/    # DocumentStore paths, Downloader, Docling Extractor
│   ├── fetcher.py    # BSE/NSE document downloader
│   ├── indexer.py    # Qdrant vector store (per-ticker collections) and Snowflake Arctic embeddings
│   ├── nifty50.py    # Constituent loader over ticker_mapping.csv
│   ├── notify.py     # One-shot completion webhook
│   ├── runlog.py     # logs/<job>/ file logging, JSONL events, counters
│   └── rag/          # Probes, retriever, reranker, and synthesizer
├── ticker_mapping.csv# Nifty 50 constituents (symbol, Screener name, sector, verified_at)
├── logs/             # Batch job logs (gitignored; GROWNXT_LOG_DIR overrides)
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
# The parity gate: import sweeps, lint, and every suite below in one command.
# Baseline is 12 of 14 -- see the status banner at the top of this file.
python scripts/parity_gate.py
python scripts/parity_gate.py --quick   # imports and lint only

# Verify reporting engine, analytics, and self-checks
python scripts/verify_reporting.py

# Verify document download, caching, and storage contracts
python scripts/verify_documents.py

# Verify Google Drive authentication and upload flow
python scripts/verify_gdrive.py

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
| `QDRANT_API_URL` | Qdrant server / Cloud URL. **Unset = in-memory store, lost on exit** (logged at ERROR) | - |
| `QDRANT_API_KEY` | Qdrant API key | - |
| `QDRANT_COLLECTION_NAME` | Shared collection name, and the prefix of per-ticker collections (`<name>_<ticker>`) | `grownxt_financial_elements` |
| `QDRANT_COLLECTION_PER_TICKER` | `0` reverts to one shared collection with a `ticker` payload filter | `1` |
| `QDRANT_PREFER_GRPC` | `0` forces the REST transport. gRPC sends vectors as binary rather than JSON; the client proves the connection at startup and falls back to REST on its own | `1` |
| `GROWNXT_LOG_DIR` | Root for batch job logs (`logs/nifty50/`) | `./logs` |
| `GROWNXT_DEVICE` | Device for Docling and the embedder: `auto`/`cuda`/`cuda:N`/`cpu`/`mps`/`xpu` | auto-detect |
| `GROWNXT_NUM_THREADS` | Extraction threads. Docling's own default is **4 on any machine** | physical cores |
| `GROWNXT_EMBED_BATCH_SIZE` | Texts per `model.encode` call | sized from VRAM |
| `GROWNXT_EMBED_FP16` | `0` keeps the embedding model in float32 | on for CUDA |
| `NIFTY50_ADMIN_TOKEN` | Bearer token for `POST /api/embeddings/nifty50/run`; unset disables the route (503) | - |
| `NIFTY50_NOTIFY_WEBHOOK_URL` | POSTed once per batch run, plus once per constituent as it finishes (`--no-ticker-notify` disables the per-ticker feed). An ntfy URL (`https://ntfy.sh/<topic>`) gets ntfy's plain-text protocol with Title/Tags/Priority headers; any other URL gets Slack/Discord/Teams-compatible JSON (`text` + counters) | - |
| `NIFTY50_NOTIFY_EMAIL_TO` | With an ntfy URL, sets the `Email` header so ntfy also forwards the notification to this inbox; set empty to disable | `shahstavan72@gmail.com` |
---

## Code & API Conventions

- **Style**: MUST follow Google Python Style Guide with strict type annotations (`typing`) and docstrings.
- **Console & Encoding**: ALWAYS ensure UTF-8 output (`cli.console_utf8()`) to avoid Windows `cp1252` encoding crashes on currency symbols (`₹`) and typography.
- **REST Endpoints**:
  - `GET /api/search?q=<query>`: Fast in-process stock lookup.
  - `GET /api/stocks/<sym>/report`: Returns metadata and Google Drive link (compiles on demand).
  - `GET /api/stocks/<sym>/report/file`: Direct binary PDF stream (`?download=1` for attachment).
  - `POST /v1/chat/completions`: OpenAI-compatible streaming chat completion proxy.
  - `POST /api/embeddings/nifty50/run`: Launches the batch as a **detached process** (never in-process) and returns `202 {run_id, status_url}`; `409` while a run is live; requires `Authorization: Bearer <NIFTY50_ADMIN_TOKEN>` (or `X-Admin-Token`). Body mirrors the CLI: `tickers`, `force`, `force_all`, `annual_reports`, `transcripts`, `presentations`, `dry_run`, plus the compute knobs `device`, `num_threads`, `embed_batch_size`, `fast_tables` (omit them to resolve from the host).
  - `GET /api/embeddings/nifty50/status[/{run_id}]`: Reads `run_manifest.json`; pure file read, safe to poll.
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
- **Per-ticker Qdrant collections**: `IndexerConfig.collection_for(ticker)` is the only way to name a collection; every search/upsert path takes the ticker. `search_batch_by_vectors` refuses a batch spanning two tickers. `output/_nifty50/` is cross-ticker state, not a stock — `index_all_tickers` skips `_`-prefixed directories.
- **`should_index_document` history**: its final comparison was inverted (re-embedding every current document, skipping real collection changes). `scripts/verify_nifty50_embeddings.py` guards it; keep that check green.
- **Layer 1 "known" means "successfully indexed by the current pipeline"**: a `doc_id` recorded as FAILED, produced by an older `CHUNKER_VERSION`/`EXTRACT_VERSION`, or living in a different collection is re-processed automatically. `Extractor.run()` itself never reads its cache — only Layer 1 prevents a redundant Docling pass.
- **`os.kill(pid, 0)` terminates the process on Windows** — use `ingestion.batch.pid_alive()` for liveness checks.
- **A CPU-only torch wheel makes every `device="auto"` mean `cpu`**, on a machine with a working GPU, with no error anywhere. `core.hardware.HardwareProfile.idle_gpu` is the detector (NVIDIA driver present, `torch.version.cuda` empty); `--hardware` surfaces it. `requirements.txt` pins the CPU wheel deliberately, for the Space.
- **The table mode is part of the extractor version, the device is not.** `extract_version(accurate_tables)` appends `+fast-tables`, so `--fast-tables` makes Layer 1 re-extract everything indexed the other way — correct, because it changes the text. Device and thread count produce the same document and must never enter that comparison, or every machine would invalidate the last one's cache.
- **`--workers` is capped to 1 under `GPU_CONCURRENCY_VRAM_GB` (8 GB)**: workers are threads sharing a process, so each puts its own layout and table models on the same card. `resolve_workers()` owns that decision and logs it.
- **Per-ticker notifications are quiet by design**: ntfy priority 2 for an OK ticker so fifty of them do not buzz a phone fifty times, 4 for `FAILED`/`PARTIAL`. The `Email` header is set **only** on the end-of-run summary -- `notify_ticker` never sets it, or one run would be fifty-one emails.
- **`Popen.pid` is not the batch process's pid.** A Windows venv `python.exe` can be a launcher stub, so `launch_detached` records the stub's id. The child therefore claims its own run via `RunManifest.claimed` rather than by comparing pids -- comparing them made every `--background` run refuse itself at startup. Keep `_check_launch_handoff` in the verification suite green.
- **The CLI entry point loads `.env` itself** (`scripts/embed_nifty50.py`). It must not rely on `ingestion.indexer` importing dotenv as a side effect: a detached run that found neither `QDRANT_API_URL` nor the webhook would embed into an in-memory store and never say so.
- **Page filtering keeps the financial tail, and fails toward keeping pages.** `ingestion/documents/sections.py` finds where an annual report's financial section starts with a cheap pypdf text pass (~60 ms/page against Docling's seconds) and converts only from there. Every uncertain case -- under 60 pages, no anchor, an unreadable or scanned PDF, an anchor on page 1 -- returns None, meaning convert everything: a dropped page is unsearchable forever, a spare one costs seconds. Measured on ADANIENT FY2026: pages 222-396, dropping 221 of 396.
- **An anchor must be the statements, not a phrase near them.** Page 191 of that filing begins "INDEPENDENT AUDITOR'S **CERTIFICATE** ON COMPLIANCE WITH THE CORPORATE GOVERNANCE REQUIREMENTS". Matching merely `independent auditor` anchors there and drags in 31 pages of the governance report the filter exists to drop, so `ANCHORS` requires the word "report". `check_page_filter` guards this; keep it green.
- **`extract_version` carries every setting that changes the text**, now the table mode *and* the page filter (`+fast-tables+fin-pages`, in that fixed order). The page range also enters `Extractor._cache_key`, so a full extraction on disk can never satisfy a filtered request or the reverse. Device and thread count still must never enter either.
- **OCR and figures are already off in the batch path** (`stages.py`, `Extractor(ocr=False, figures=False)`). The `ocr: bool = True` in `ingestion/documents/extract.py` is only the dataclass default and does not apply to a Nifty 50 run -- do not go looking for a saving there.
- **The embedding matrix is never materialised in Python.** `index_chunk_set` calls `embed_matrix` and hands the array to `upload_vectors`, which uses Qdrant's `upload_collection`. Building `PointStruct`s instead means one Python float per dimension per chunk -- some 2.4 million objects for one annual report. `embed_texts` still returns lists for the search path.
- **Only the last upsert batch waits.** Qdrant applies a shard's operations in order, so an acknowledged final batch implies the ones before it -- which preserves the invariant that `state.json` records INDEXED only once the server holds the vectors. Waiting on all of them cost a round trip per 128 points.
- **Batch logs** are in `logs/nifty50/runs/<run_id>/events.jsonl` (one JSON object per stage boundary: catalogue counts, bytes downloaded, pages/tables/figures extracted, chunks, embed rate, Qdrant point counts). Query with `jq`, not by reading `run.log`.

---

## Definition of Done

1. **Implementation**: Validated against Google Python Style Guide and type checked.
2. **Verification**: Relevant verification suite in `scripts/verify_*.py` passes with zero failures.
3. **Compilation**: Typst compiles report without syntax errors (`python scripts/generate_report.py <TICKER>`).
4. **Security & Deployment**: No tokens or credentials committed; Docker container and Hugging Face Space (`app.py`) build and boot cleanly on port 7860.
5. **Contract**: API route compatibility and JSON schemas preserved.
