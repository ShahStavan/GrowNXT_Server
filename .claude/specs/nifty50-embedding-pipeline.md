# Spec: Nifty 50 Batch Embedding Pipeline with Incremental State

- **Status**: Draft, for review
- **Author**: Claude (via /specs), for @jenish.gajera
- **Date**: 2026-09-02
- **Target branch**: `feat/tier2-composites-and-verification` (or a new `feat/nifty50-embeddings`)

## 1. Goal

Add a single script that embeds the corporate filings (annual reports, concall
transcripts, concall presentations) for all 50 Nifty 50 constituents into the
Qdrant vector store used by Track 2 (Qualitative RAG), and that is **safe to
re-run on a schedule**: a second run against a ticker whose filings have not
changed must do no network I/O beyond one catalogue check, no Docling
extraction, and no re-embedding. A run only does work for a ticker when the
issuer has published something new — a fresh annual report, a new quarter's
transcript, or a new investor presentation.

This is an orchestration layer. It does not change what a chunk is or how it
is embedded; it wires together modules that already exist and already do
their own incremental caching, and it adds the one piece that is missing:
something that runs that wiring across all 50 tickers with its own top-level
state.

The batch job must also be **asynchronous and non-blocking**: starting it
must return control to whoever started it (a shell prompt or an API caller)
immediately, not hold it for the hours a cold 50-ticker run can take (§6).
And it must **notify once, when the entire batch has finished** — every
ticker in scope attempted and every successfully-chunked document confirmed
persisted in Qdrant — rather than leaving the operator to poll a log file to
find out. Both requirements are addressed in §3.8 and §3.9.

## 2. Background — what already exists

The ingestion pipeline (`ingestion/`) already implements, and mostly already
caches, every stage:

```
catalog.fetch_catalog(ticker)           -> Catalog (list of CatalogEntry: doc_id, doc_type, source_url)
documents.download.Downloader.fetch_all -> downloads to output/<T>/documents/<doc_id>.pdf, reuses an existing file on disk unless force=True
documents.extract.Extractor.run         -> Docling parse, caches to output/<T>/extracted/<doc_id>.json (always re-parses when called; see §3.2)
chunker.chunk_document                  -> ChunkSet with a content fingerprint (sha256 over chunk text), written to output/<T>/chunks/<doc_id>.json
indexer.QdrantVectorIndexer.index_ticker_documents -> embeds chunk files into Qdrant, skipping any doc_id whose fingerprint/model/collection already matches output/<T>/state.json
```

Key files, for reference:
- `ingestion/catalog.py` — `fetch_catalog()`. `doc_id` is derived from the
  filing's **period** (`annual_report_FY2025`, `transcript_2025_07`, …), not
  from the URL, so a newly published filing always gets a doc_id that has
  never been seen before, and an already-ingested filing keeps the same
  doc_id even if the issuer re-links it. This is the fact the whole
  change-detection design in §3 leans on.
- `ingestion/documents/storage.py` — `DocumentStore`, the path layout under
  `output/<TICKER>/`.
- `ingestion/documents/download.py` — `Downloader`. Already skips a
  re-download when a valid PDF for that `doc_id` is on disk (`fetch()`,
  line ~239), so calling it repeatedly across runs is free.
- `ingestion/documents/extract.py` — `Extractor.run()`. **Does not check for
  an existing cached extraction before running Docling** — every call
  re-parses the PDF (see §3.2 for why this matters and how the new script
  works around it).
- `ingestion/chunker.py` — `chunk_document()`, `write_chunk_cache()`,
  `read_chunk_cache()`. `ChunkSet.fingerprint` is a sha256 over every chunk's
  id+text, computed automatically inside `chunk_document()`.
- `ingestion/indexer.py` — `QdrantVectorIndexer`. Owns `output/<TICKER>/state.json`
  (`TickerState` → `{doc_id: DocumentIndexState(fingerprint, embedding_model,
  qdrant_collection, indexed_at, status)}`) and `should_index_document()`
  (line 371), which already implements exactly the "re-embed only on change"
  rule at the single-document level: skip when the fingerprint, embedding
  model, and collection all still match what's recorded.
- `ingestion/rag/pipeline.py::_auto_ingest_and_index` (line ~135) is the one
  place today that runs catalog → download → extract → chunk → index
  end-to-end, but only as a bootstrap fallback inside the RAG pipeline: it
  fires only when a ticker has **zero** documents on disk, and it hard-caps
  itself to the single latest annual report, 3 transcripts, and 1
  presentation. It is not reachable as a standalone batch tool and never
  re-checks a ticker that already has *something* ingested.

### 2.1 Dead code — do not build on these

Two scripts documented in `CLAUDE.md` do not run against the current
codebase; they import names that no longer exist (from a pre-Docling
refactor):

- `scripts/ingest_documents.py` imports `DEFAULT_ANNUAL_SKIP_SECTIONS` from
  `ingestion.chunker`, and `DocumentRegistry` / `IngestionPipeline` /
  `DEFAULT_EMBED_MODEL` with no corresponding import at all — none of these
  four names exist anywhere in the repo. Running it raises `ImportError`
  immediately.
- `scripts/verify_ingestion.py` imports `ingestion.layout`, `ingestion.parsers`,
  `ingestion.embedder`, `ingestion.prompts` — none of these modules exist;
  the current package is `ingestion/documents/`, `ingestion/chunker.py`,
  `ingestion/indexer.py`.

The new script must be built directly on `ingestion/catalog.py`,
`ingestion/documents/*`, `ingestion/chunker.py`, and `ingestion/indexer.py` —
the same modules `ingestion/rag/pipeline.py` actually uses — not on either of
the two scripts above. This should be called out to the user separately;
it's a pre-existing bug, not something this spec fixes, but it explains why
"just extend `ingest_documents.py`" is not a viable starting point.

### 2.2 Blocking issue — Qdrant persistence

`IndexerConfig.from_env()` / `QdrantVectorIndexer.client` (indexer.py:198-213)
falls back to `QdrantClient(location=":memory:")` whenever `QDRANT_CLUSTER` /
`QDRANT_HOST` / `QDRANT_URL` is unset, and no such variable is documented in
`.env.example` or `CLAUDE.md`. An in-memory client's collection is lost the
moment the Python process exits. Batch-embedding all 50 Nifty constituents
into a store that evaporates on exit would make the state-tracking in this
spec pointless — `state.json` would faithfully record "INDEXED", but the
vectors it refers to would already be gone. **This must be resolved before
the new script is useful**, either by:

- (a) pointing `QDRANT_URL`/`QDRANT_API_KEY` at a persistent Qdrant Cloud
  cluster (free tier is enough for 50 tickers' worth of chunks), or
- (b) adding on-disk persistence to `IndexerConfig`/`QdrantVectorIndexer` via
  `QdrantClient(path=...)` (Qdrant's embedded/local mode — no server needed),
  gated by a new `QDRANT_LOCAL_PATH` env var, so the platform keeps working
  with zero external services for local/dev use.

This spec assumes **(b)** is added as a small, additive change to
`ingestion/indexer.py` (new `qdrant_path: Optional[str]` field on
`IndexerConfig`, checked before the `:memory:` fallback), since it requires
no new account/credentials and matches "no database engine" / file-backed
philosophy already stated in `CLAUDE.md`. (a) remains available for anyone
who sets the existing env vars — local path is only the fallback default the
same way `:memory:` is today, just a useful one.

## 3. Design

### 3.1 Two-layer change detection

**Layer 1 — catalogue diff (new, in this script).** Once per ticker per run,
fetch the catalogue and diff its `doc_id`s against the ones this ticker has
already fully processed (`output/<TICKER>/state.json`'s `documents` keys,
via `QdrantVectorIndexer.load_ticker_state()`). Only `doc_id`s that are
**absent from that state** are downloaded/extracted/chunked at all. Since
`doc_id` is period-derived (§2), an issuer publishing FY26's annual report
produces a `doc_id` (`annual_report_FY2026`) that was never in last run's
catalogue, so it is picked up automatically; FY25's report, already indexed,
is never touched again. This is what makes a steady-state run cheap: no
Docling, no embedding calls, just one catalogue HTTP request per ticker.

**Layer 2 — content fingerprint (existing, unchanged).**
`QdrantVectorIndexer.index_chunk_set()` / `should_index_document()` already
guard the embedding step itself by comparing `ChunkSet.fingerprint` against
what's recorded. This stays as the safety net for the case Layer 1 can't see
— an issuer replacing the file behind an *existing* `doc_id` (e.g. a
corrected filing) without a new period. Layer 1 alone would skip it since
the `doc_id` isn't new; Layer 2 alone would still recompute a fresh Docling
parse it doesn't need to when nothing changed. Layer 1 is the cost
optimization, Layer 2 is the correctness guarantee — the new script always
calls `index_ticker_documents()` at the end regardless of what Layer 1 found,
so Layer 2 keeps working exactly as it does today.

`--force` (see §3.4) bypasses Layer 1 for named tickers so a corrected filing
can still be picked up on demand without waiting for a period change.

### 3.2 Why extraction needs its own guard

`Extractor.run()` has no cache check of its own — it re-runs Docling every
time it's called, and a 100+ page annual report is minutes of CPU (see
`ingestion/documents/extract.py` module docstring). Layer 1 already prevents
this in steady state by never calling `Extractor.run()` for a `doc_id`
already in `state.json`. The new script must not call extract/chunk itself
for any `doc_id` that Layer 1 says is already known — it should go straight
from "already known" to "included in the `index_ticker_documents()` call",
where Layer 2's fingerprint check makes that call a no-op read of
`chunks/<doc_id>.json` plus a state comparison, nothing more.

### 3.3 New data: Nifty 50 constituent list

There is no endpoint in the Financial Data Collector service (`api/search.py`,
`ingestion/catalog.py`) that returns index membership — only free-text
company search. Nifty 50 constituents also change twice a year (NSE's March
and September semi-annual reviews), so this cannot be inferred from anything
already in the repo and must ship as maintained data, not be hardcoded in
the script:

- **New file**: `ingestion/data/nifty50.json`
  ```json
  {
    "index": "NIFTY 50",
    "as_of": "<date this list was last checked against the official NSE factsheet>",
    "source": "https://www.nseindia.com/products-services/indices-nifty50-index (verify at implementation time)",
    "constituents": [
      {"symbol": "RELIANCE", "name": "Reliance Industries Ltd."},
      {"symbol": "TCS", "name": "Tata Consultancy Services Ltd."}
    ]
  }
  ```
- `symbol` must be the exact string the Financial Data Collector service
  expects (matches what `ingestion/catalog.py::fetch_catalog` and
  `core.config.safe_ticker` already handle, including ampersand tickers like
  `M&M`, `BAJAJ-AUTO`).
- **This spec intentionally does not hardcode the 50 symbols.** My training
  data is stale relative to today's date (2026-09-02) and NSE has run at
  least one, likely two, semi-annual rebalances since. Shipping a guessed
  list as fact would silently mis-embed the corpus. **Action item before
  implementation**: pull the current constituent list from NSE's live
  Nifty 50 factsheet/CSV (or ask the user to paste it) and populate this
  file as the first implementation step, not from memory.
- A loader helper, `ingestion/data/__init__.py::load_nifty50()`, returns
  `list[str]` of symbols, with an optional `path` override — this is the one
  function the new script and any future caller both use, so the list is
  never duplicated inline.

### 3.4 New script: `scripts/embed_nifty50.py`

CLI, following the argparse/`scripts/cli.py` conventions already used by
`scripts/ingest_documents.py` and `scripts/generate_report.py`.

```
python -m scripts.embed_nifty50                      # all 50, incremental
python -m scripts.embed_nifty50 --tickers TCS INFY    # just these
python -m scripts.embed_nifty50 --force TCS           # ignore Layer 1 for TCS only
python -m scripts.embed_nifty50 --force-all           # ignore Layer 1 for every ticker
python -m scripts.embed_nifty50 --status              # read-only coverage report, no network
python -m scripts.embed_nifty50 --status --json
python -m scripts.embed_nifty50 --limit 5             # smoke test: first 5 tickers only
python -m scripts.embed_nifty50 --dry-run             # catalogue + diff only, no download/extract/embed
```

Arguments:

| Flag | Meaning |
|---|---|
| `--tickers SYM [SYM ...]` | Restrict to these symbols instead of the full Nifty 50 list |
| `--nifty50-file PATH` | Override `ingestion/data/nifty50.json` |
| `--annual-reports N` | Years of annual reports to request per ticker (default 1) |
| `--concall-years N` | Years of concalls to request per ticker (default 1) |
| `--force [SYM ...]` | Re-run Layer 1 (and therefore extract+chunk) for named tickers even if their doc_ids are already known; no value = same as `--force-all` |
| `--force-all` | Re-run Layer 1 for every ticker in scope |
| `--limit N` | Process only the first N tickers (after `--tickers` filtering) — smoke testing |
| `--workers N` | Tickers processed concurrently (default 1; see §3.5) |
| `--dry-run` | Fetch catalogues and print the diff against known state; no download/extract/chunk/embed |
| `--status` | Skip all network calls; read every ticker's `output/<TICKER>/state.json` and print coverage |
| `--json` | Machine-readable summary on stdout instead of the text table |
| `--continue-on-error` | Default true; one ticker's failure does not stop the batch (matches the existing "one bad filing/download doesn't abort the rest" pattern in `download.py`/`pipeline.py`) |
| `--background` / `--detach` | Launch the run as a detached background process and return immediately (see §3.8) instead of blocking the invoking shell |
| `--run-id ID` | Explicit id for this run's manifest entry (default: generated timestamp); lets a caller poll `--status --run-id ID` for the run it just started |
| `--notify-webhook URL` | Overrides `NIFTY50_NOTIFY_WEBHOOK_URL` for this run (see §3.9) |
| `--no-notify` | Suppress the completion notification for this run (manifest/log status is still written) |
| `--verbose` | Debug logging |

Per-ticker algorithm:

```
for ticker in scope:
    catalog = fetch_catalog(ticker, annual_reports, concall_years)   # Layer 1 input
    state   = QdrantVectorIndexer.load_ticker_state(ticker)          # what's already embedded
    known_ids = set(state.documents.keys())
    pending = [entry for entry in catalog.entries
               if entry.doc_id not in known_ids or ticker in forced]

    if dry_run:
        record diff (new=len(pending), unchanged=len(catalog.entries) - len(pending))
        continue

    store = DocumentStore.open(ticker).ensure()
    downloader = Downloader(store=store)
    results = downloader.fetch_all([DownloadRequest(doc_id=e.doc_id, url=e.source_url, label=e.label)
                                     for e in pending])

    extractor = Extractor(ocr=False, figures=False)
    for r in results:
        if not r.ok: record failure(doc_id, "download", r.error); continue
        entry = lookup(pending, r.doc_id)
        try:
            doc = extractor.run(pdf=r.path, store=store, doc_id=r.doc_id,
                                 doc_type=entry.doc_type, ticker=ticker, label=entry.label, write=True)
            chunk_set = chunk_document(doc, chunk_size=800, chunk_overlap=100)
            write_chunk_cache(chunk_set, store.root / "chunks" / f"{r.doc_id}.json")
        except Exception as exc:
            record failure(doc_id, "extract_or_chunk", exc); continue

    # Layer 2 runs regardless of whether Layer 1 found anything new —
    # this is also what turns a previously-FAILED doc_id back to INDEXED
    # once its chunk file exists, and is a cheap no-op when nothing changed.
    index_results = QdrantVectorIndexer().index_ticker_documents(ticker, force=ticker in forced)
    record per-ticker summary (new, unchanged, failed, chunks_indexed, elapsed)
```

Note the `pending` selection also re-attempts any `doc_id` that is in the
catalogue but was previously recorded with `status != "INDEXED"` in
`state.json` (a prior run's download or extraction failure) — Layer 1's
"already known" check should mean "already **successfully** indexed", not
merely "seen before", or a transient failure would silently become
permanent.

### 3.5 Concurrency & politeness

- **Within a ticker**: unchanged — `Downloader` already runs its own
  thread pool (`DEFAULT_WORKERS = 4`) and per-host backoff.
- **Across tickers**: default `--workers 1` (sequential). Docling extraction
  is CPU-bound and the embedding model likely runs on the same CPU
  (`core/llm_config.py` / `ingestion/indexer.py`'s `SentenceTransformer`
  device selection); running several tickers' Docling conversions
  concurrently would thrash rather than speed anything up on typical
  hardware. `--workers` is exposed as an escape hatch for a machine known to
  have GPU/many cores, capped at a small number (e.g. 4) rather than left
  unbounded, matching the existing `DEFAULT_WORKERS` philosophy.
- A short fixed delay between tickers' catalogue requests (e.g. 1–2s) is
  worth adding since a full run hits the Financial Data Collector service's
  `/documents` endpoint 50 times in a row; the existing `fetch_catalog()`
  retry/backoff (`ingestion/catalog.py:213-276`) already covers transient
  failures, this is purely about not hammering a shared free service.

### 3.6 Top-level run manifest

In addition to the per-ticker `output/<TICKER>/state.json` that
`QdrantVectorIndexer` already owns (unchanged), the script writes one
cross-ticker manifest for operational visibility — "did last night's run
work, and on which tickers" — since nothing today aggregates 50 tickers'
worth of `state.json` into one view:

**New file**: `output/_nifty50/run_manifest.json`

```json
{
  "schema_version": 1,
  "nifty50_source": "ingestion/data/nifty50.json",
  "nifty50_as_of": "2026-08-01",
  "run_id": "20260902T040000Z",
  "pid": 18452,
  "status": "COMPLETED_WITH_ERRORS",
  "total_tickers": 50,
  "completed_tickers": 50,
  "last_run_started_at": "2026-09-02T04:00:00Z",
  "last_run_finished_at": "2026-09-02T05:12:33Z",
  "notified_at": "2026-09-02T05:12:34Z",
  "embedding_model": "snowflake-arctic-embed-m-v1.5",
  "qdrant_collection": "grownxt_financial_elements",
  "tickers": {
    "TCS": {
      "last_attempted_at": "2026-09-02T04:03:11Z",
      "last_success_at": "2026-09-02T04:03:11Z",
      "catalog_entries": 5,
      "new_documents": 0,
      "unchanged_documents": 5,
      "failed_documents": 0,
      "status": "OK",
      "errors": []
    },
    "M_M": {
      "last_attempted_at": "2026-09-02T04:04:02Z",
      "last_success_at": "2026-08-15T04:00:00Z",
      "catalog_entries": 6,
      "new_documents": 1,
      "unchanged_documents": 4,
      "failed_documents": 1,
      "status": "PARTIAL",
      "errors": [{"doc_id": "transcript_2026_07", "stage": "download", "error": "HTTP 503"}]
    }
  }
}
```

This is written atomically (temp file + `os.replace`, matching the existing
pattern in `ingestion/indexer.py::save_ticker_state`), and is what
`--status`/`--json` and `GET /api/embeddings/nifty50/status` (§3.8) read for
a network-free report. It is *derived, regenerable* state — it can always be
rebuilt from the 50 `state.json` files plus a fresh catalogue fetch — so
losing it is not data loss, only a loss of run history/timing.

The top-level `status` field (`RUNNING` while in progress; `COMPLETED` when
every ticker succeeded; `COMPLETED_WITH_ERRORS` when the run finished but at
least one ticker had a failed document; `FAILED` only if the run aborted
before attempting every ticker, e.g. a crash) is updated as the run
progresses, and `completed_tickers`/`total_tickers` is what §3.8's polling
and progress reporting are driven from. `notified_at` is set once the
notifier in §3.9 has fired, so a crashed/restarted status poller can tell
whether the one-time notification already went out.

### 3.7 Output

Text mode mirrors the existing `_grid()` table style in
`scripts/ingest_documents.py` (`doc_id | status | chunks | model` per row,
one table per ticker) plus a closing 50-row summary table (`ticker |
new | unchanged | failed | status`) and a final line: `"N/50 tickers OK, M
partial, K failed."` Exit code is non-zero iff any ticker has a failed
document and `--continue-on-error` still ran the rest — same convention as
`scripts/ingest_documents.py::main`.

### 3.8 Asynchronous, non-blocking execution

The orchestration logic in §3.4 (`run_batch(tickers, ...) -> RunSummary`)
must live in one plain function that every entry point below calls — the CLI
already described, a background process, and an API endpoint are three ways
to *start* the same work, not three implementations of it.

**Why not an in-process `asyncio` task or FastAPI `BackgroundTasks`.**
`app.py` already runs one FastAPI/Gradio process serving report generation
and SLM chat. Docling extraction and `SentenceTransformer.encode()` are both
CPU-bound, synchronous, GIL-holding calls (there is no `await` anywhere
inside them) — scheduling the batch job as an `asyncio` task or a FastAPI
`BackgroundTasks` callback would still run it on the same process's CPU and
starve every other request for the hours a cold run takes, which defeats
"non-blocking" in practice even though the HTTP call returns instantly.
"Non-blocking" here has to mean a **separate OS process**, not just a
separate coroutine.

**Mode 1 — CLI, detached process (`--background`).** When passed, the CLI
does not run the batch itself; it spawns a second, fully independent process
running the same module with `--background` stripped out
(`subprocess.Popen([sys.executable, "-m", "scripts.embed_nifty50", ...],
creationflags=subprocess.DETACHED_PROCESS` on Windows /
`start_new_session=True` on POSIX, stdout/stderr redirected to
`output/_nifty50/runs/<run_id>.log`), writes that PID and `run_id` into
`run_manifest.json` as `status: "RUNNING"`, prints the `run_id` and log path,
and returns immediately (exit code 0 means "started", not "finished").
`--status --run-id ID` (or plain `--status`, which shows the most recent run)
polls the manifest, and `--tail` / reading the log file gives live progress —
this reuses the manifest design from §3.6 rather than adding a second
mechanism.

**Mode 2 — API trigger, for programmatic/UI callers.** Two new FastAPI
routes on the existing app (`app.py`, alongside `/api/stocks/...`):

- `POST /api/embeddings/nifty50/run` — body optionally carries `tickers`,
  `force`, `annual_reports`, `concall_years` (same semantics as the CLI
  flags). The handler validates input, then launches the batch exactly the
  way Mode 1 does — a detached `subprocess.Popen` of
  `python -m scripts.embed_nifty50`, never an in-process call — and returns
  `202 Accepted` with `{"run_id": ..., "status_url": "/api/embeddings/nifty50/status/<run_id>"}`
  immediately. Returns `409 Conflict` if a run is already `RUNNING`
  (single-flight: 50 tickers' worth of Docling is not something to run
  twice concurrently on one box; matches §3.5's concurrency caution).
- `GET /api/embeddings/nifty50/status/{run_id}` (and `GET
  /api/embeddings/nifty50/status` for the latest) — reads `run_manifest.json`
  and returns it as JSON: overall `status`, per-ticker breakdown, and
  progress (`completed_tickers / total_tickers`). Pure file read, no
  subprocess involved, so this is always fast and safe to poll frequently.

Both modes write to and read from the same `run_manifest.json`, so a run
started from the CLI can be polled from the API and vice versa.

### 3.9 Completion notification

A notifier fires **exactly once per run**, after the last ticker in scope has
been attempted *and* `index_ticker_documents()` has returned for it — i.e.
after Qdrant has actually acknowledged the upserts (`upsert_points(...,
wait=True)` in `indexer.py:614-618` already blocks until Qdrant confirms
each batch), not merely after local chunk files are written. This directly
answers "notify once all embeddings are generated **and saved** on the
Qdrant vector store" — the notification is gated on Qdrant's own
acknowledgment, not on local disk state.

The notifier is a small pluggable component (`scripts/embed_nifty50.py` or a
new `ingestion/notify.py`), env-gated the same way Google Drive delivery is
optional in `storage/gdrive.py` — every channel degrades to "log only" when
unconfigured, so the batch job never fails or blocks because a webhook is
unreachable:

| Channel | Always on? | Trigger / config |
|---|---|---|
| Structured log line + `run_manifest.json` status | Always | `status` set to `COMPLETED`, `COMPLETED_WITH_ERRORS`, or `FAILED`; this alone is enough for `--status`/`GET .../status` polling to see completion |
| Generic webhook (Slack-compatible incoming webhook, Discord, MS Teams, or a custom endpoint) | Optional | `NIFTY50_NOTIFY_WEBHOOK_URL` env var, or `--notify-webhook`; a single `POST` of a JSON summary (counts, elapsed, failed tickers/doc_ids, `run_id`) on completion, best-effort with one retry — a failed webhook delivery is logged, never raised |
| Email | Optional | `NIFTY50_NOTIFY_EMAIL_TO` + standard `SMTP_HOST`/`SMTP_PORT`/`SMTP_USER`/`SMTP_PASSWORD` env vars; only attempted when `NIFTY50_NOTIFY_EMAIL_TO` is set |

A partial or failed run notifies too, with a distinct subject/summary line
("47/50 OK, 3 failed" vs. "50/50 OK") — silence is only acceptable for a
run still in progress, never for one that ended badly. `--no-notify`
suppresses the webhook/email channels for a single invocation (e.g. smoke
tests) without touching the always-on manifest/log status.

## 4. Non-goals

- Scheduling *when* the batch runs (cron / Windows Task Scheduler / GitHub
  Actions) — out of scope for this spec. §3.8 makes the job safe to trigger
  from a scheduler later (it's just another caller of the CLI or the API),
  but wiring one up is a separate task.
- A real task queue (Celery, RQ, Redis, etc.) — the "non-blocking" and
  "notify on completion" requirements are met with a detached OS process
  plus a JSON manifest (§3.8, §3.9), consistent with the project's existing
  no-external-database, file-backed philosophy (`CLAUDE.md`: "No Database
  Engine"). Introducing a broker would be disproportionate to running one
  batch job.
- Fixing `scripts/ingest_documents.py` or `scripts/verify_ingestion.py` —
  flagged in §2.1 as pre-existing breakage, not touched here.
- Changing chunk size, embedding model, or how many years of filings are
  fetched by default — this spec reuses the existing defaults
  (`chunk_size=800`, `chunk_overlap=100`, `annual_reports=1`,
  `concall_years=1`) from `ingestion/rag/pipeline.py`'s bootstrap path.
- A dynamic/live Nifty 50 membership lookup — no such upstream endpoint
  exists today (§3.3); the list is static data that needs periodic manual
  refresh.
- A UI for triggering or watching runs — §3.8 adds an API surface a future
  UI could call, but building that UI is not part of this spec.

## 5. Verification plan

New `scripts/verify_nifty50_embeddings.py`, following the
`scripts/checks.py` `Report`/`banner` harness already used by
`scripts/verify_reporting.py` and `scripts/verify_gdrive.py` (not
`verify_ingestion.py` — see §2.1):

**Offline checks (no network, run in CI/pre-commit):**
- `ingestion/data/nifty50.json` parses, has exactly 50 unique symbols, and
  every symbol round-trips through `core.config.safe_ticker` without
  collision (two different symbols must not fold to the same directory name).
- Given a synthetic `Catalog` and a synthetic `TickerState`, the Layer-1
  pending-set calculation returns exactly the entries whose `doc_id` is
  absent or whose recorded `status != "INDEXED"` — this is the core logic
  this spec adds, so it's the one thing worth a real unit test rather than
  an end-to-end run.
- `run_manifest.json` round-trips (`to_dict`/`from_dict`) and a corrupt/
  missing manifest degrades to "no history" rather than raising, matching
  `QdrantVectorIndexer.load_ticker_state`'s own corrupt-file handling
  (indexer.py:344-349).
- `--status` produces output using only files already on disk (mock
  `output/` with 2-3 fake ticker directories; assert no `requests` call is
  made — can reuse `monkeypatch`/a stub for `fetch_catalog`).
- `--background` returns within, say, 2 seconds and prints a `run_id`,
  regardless of how long the spawned process will actually take (stub the
  subprocess launch in the test; assert `Popen` was called and the parent
  process's own runtime stayed short).
- The notifier is called exactly once per run and only after the run-level
  `status` transitions out of `RUNNING` (feed it a fake `RunSummary` and
  assert one webhook POST, not one per ticker); a webhook that raises/times
  out is caught and logged, and does not change the run's recorded `status`.

**Live checks (opt-in, one real ticker, e.g. `python -m
scripts.verify_nifty50_embeddings --live TCS`):**
- Run the script once for one ticker: expect ≥1 document `INDEXED`.
- Run it again immediately: expect zero new downloads, zero Docling calls,
  zero embedding calls, and `status: "OK"` with `new_documents: 0` — this is
  the acceptance criterion from the goal in §1, checked directly the same
  way `verify_ingestion.py`'s docstring describes (run twice, assert the
  second run's own counters), just against the modules that actually exist.
- Confirm the collection survives a fresh Python process (`QdrantClient`
  reconnect + `count()` on the collection) once §2.2 is resolved — this is
  the check that would have caught the in-memory-Qdrant problem.
- Trigger a real `--background` run for one ticker, confirm the CLI returns
  before the run finishes, then poll `--status --run-id ID` until `status`
  leaves `RUNNING`, and confirm a webhook receiver (or a captured `--notify-webhook`
  pointed at a local test server) received exactly one POST, timestamped
  after Qdrant's `count()` for that ticker's points had already increased.

## 6. Operational notes

- **Runtime**: Docling extraction is "minutes of CPU" per large annual
  report (per `ingestion/documents/extract.py` docstring); a full cold run
  across 50 tickers × ~5 documents each is realistically hours, not minutes.
  Design for unattended, resumable execution (kill and re-run picks up where
  it left off, per §3.1) rather than a single fast run.
- **Steady-state runtime** (nothing new published) should be roughly
  50 × (one catalogue HTTP call + one local `state.json` read) — seconds,
  not hours. This is the number worth watching to confirm the incremental
  design is actually working in practice.
- **Disk**: `output/<TICKER>/{documents,extracted,figures,chunks}/` for 50
  tickers will be materially larger than today's single/handful-of-ticker
  footprint. No cap is proposed here since `CLAUDE.md` already documents
  `OUTPUT_DIR` as overridable per-deployment.
- **Qdrant collection size**: 50 tickers × ~5 filings × (a few hundred chunks
  each, per existing single-ticker runs) is on the order of tens of
  thousands of points at 768 dimensions — comfortably within a free-tier
  Qdrant Cloud cluster or local on-disk storage.
- **Process lifetime for `--background`**: a detached process survives the
  parent shell/API worker exiting, but not the machine rebooting or the
  Hugging Face Space restarting. On a Space (ephemeral filesystem between
  restarts), a long background run interrupted by a redeploy simply resumes
  cleanly on the next invocation thanks to §3.1's per-`doc_id` state — there
  is nothing to reconcile, it just re-fetches catalogues and finds most
  `doc_id`s already known. This is a case worth calling out precisely
  because it's already handled for free by the incremental design, not
  because it needs new code.
- **Single-flight enforcement**: both the CLI's `--background` and the API's
  `POST .../run` must refuse to start a second run while `run_manifest.json`
  says `status: "RUNNING"` for an existing `pid` that is still alive
  (checked with `os.kill(pid, 0)`/`psutil`, falling back to "assume stale
  after N hours" if the PID check isn't portable enough to trust) — 50
  tickers of concurrent Docling work on one box is a resource fight, not a
  speedup (§3.5).

## 7. Open questions for the user

1. Persistent Qdrant: local on-disk (`QDRANT_LOCAL_PATH`, no external
   account) or a Qdrant Cloud cluster (needs `QDRANT_URL`/`QDRANT_API_KEY`)?
   §2.2 assumes local on-disk as the default with cloud still supported via
   existing env vars.
2. Confirm `annual_reports=1, concall_years=1` as the default depth for the
   initial Nifty 50 backfill, or a deeper history (e.g. 3 years) for the
   first run only.
3. Where should the authoritative Nifty 50 list be sourced from at
   implementation time — should it be pasted in by the user, or should the
   agent fetch NSE's public constituent CSV at that point?
4. Which completion-notification channel(s) should actually be wired up
   first — the manifest/log status (§3.9) is always on and requires no
   credentials; webhook and email are both optional and additive. Is a
   webhook (e.g. Slack) enough, or is email delivery required from day one?
5. Should the API trigger (`POST /api/embeddings/nifty50/run`, §3.8) be
   public on the deployed Hugging Face Space, or gated behind some auth —
   today's `/api/*` routes in `app.py` have none, and triggering hours of
   CPU work is a more sensitive action than generating one report.

## 8. Implementation checklist

- [ ] Resolve NSE Nifty 50 constituent list (do not use a memorized list —
      verify against a live source per §3.3) and add `ingestion/data/nifty50.json`
      + `ingestion/data/__init__.py::load_nifty50()`.
- [ ] Add `qdrant_path` / `QDRANT_LOCAL_PATH` persistent on-disk option to
      `IndexerConfig`/`QdrantVectorIndexer.client` in `ingestion/indexer.py`.
- [ ] Add `scripts/embed_nifty50.py` implementing §3.4–3.7, with the shared
      `run_batch()` function §3.8 requires every entry point to call.
- [ ] Add `--background`/`--detach` detached-process launch and `--status
      --run-id` polling to `scripts/embed_nifty50.py` (§3.8).
- [ ] Add `POST /api/embeddings/nifty50/run` and `GET
      /api/embeddings/nifty50/status[/{run_id}]` to `app.py`, both backed by
      the same detached-subprocess launch and manifest read as the CLI
      (§3.8) — resolve open question 5 (auth) before exposing this publicly.
- [ ] Add the pluggable notifier (log/manifest always-on, webhook + email
      optional) and wire it to fire once at run completion (§3.9).
- [ ] Add `scripts/verify_nifty50_embeddings.py` implementing §5, including
      the background-mode and notifier checks.
- [ ] Document the new script, `QDRANT_LOCAL_PATH`, the new API routes, and
      the `NIFTY50_NOTIFY_*`/`SMTP_*` env vars in `CLAUDE.md`'s Development
      Commands, REST Endpoints, and Configuration tables.
- [ ] Do a `--limit 3 --dry-run` smoke test, then a real `--limit 3` run,
      then a second `--limit 3` run to confirm the steady-state no-op
      behavior, then a `--limit 3 --background` run to confirm it returns
      immediately and the notifier still fires, before running against all
      50.
