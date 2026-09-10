# Plan: Nifty 50 Batch Embedding Pipeline

- **Status**: **Implemented** (2026-09-03). Decisions taken: **B.1** existing Qdrant
  server via `QDRANT_API_URL`/`QDRANT_API_KEY`/`QDRANT_COLLECTION_NAME` (read from
  the environment; the indexer now recognises these names — it previously only
  read `QDRANT_URL`, so the `.env` values were silently ignored and it fell back
  to `:memory:`). **B.2** constituents from `ticker_mapping.csv` via
  `ingestion/nifty50.py` (no separate `nifty50.json`). **B.3** newest 1 annual
  report + 1 transcript + 1 presentation per ticker (`--annual-reports/--transcripts/--presentations`).
  **B.4** log/manifest always on + optional webhook; `POST .../run` gated by
  `NIFTY50_ADMIN_TOKEN`. Verification: `scripts/verify_nifty50_embeddings.py`
  (77 offline checks; `--live SYM` for the two-run no-op test).
- **Original status**: Ready for implementation, pending the 4 decisions in §B
- **Spec**: [`.claude/specs/nifty50-embedding-pipeline.md`](../specs/nifty50-embedding-pipeline.md)
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-03
- **Target branch**: `feat/nifty50-embeddings` (cut from `feat/tier2-composites-and-verification`)

---

## A. Spec verification — what I checked and what it missed

I re-read the spec against the actual code before planning. The spec's core
design (two-layer change detection, period-derived `doc_id`s, per-ticker
`state.json`, detached-process async, run manifest) is **sound and confirmed
accurate**. These claims all verified true:

| Spec claim | Verified |
| :--- | :--- |
| `doc_id` is period-derived, not URL-derived | ✅ `ingestion/catalog.py:144-192` — `annual_report_<FY>`, `transcript_<YYYY_MM>`, `presentation_<YYYY_MM>` |
| `Downloader.fetch()` reuses an on-disk PDF | ✅ `ingestion/documents/download.py:252-254` (`_reuse` guard, skipped only on `force=True`) |
| `Extractor.run()` has **no** cache read — always re-parses | ✅ `ingestion/documents/extract.py:227-297` writes via `write_extraction` but never checks `store.extraction(doc_id).exists()` |
| `ChunkSet.fingerprint` is auto-computed sha256 | ✅ `ingestion/chunker.py:589-599` (over chunk id+text **plus** `params`) |
| `scripts/ingest_documents.py` + `scripts/verify_ingestion.py` are dead code | ✅ Import names that exist nowhere in the repo |
| No `QDRANT_*` var is documented anywhere | ✅ `QDRANT_*` appears only in `ingestion/indexer.py`, `requirements*`, `README.md` — **not** in `.env.example`, `docker-compose.yml`, `Dockerfile`, or `CLAUDE.md` |
| `upsert_points(..., wait=True)` blocks on Qdrant ack | ✅ `ingestion/indexer.py:633-637` |

Below are the things the spec is **wrong about or does not cover**. Several are
blocking — F1 in particular defeats the spec's own §1 acceptance criterion.

---

### F1 — 🔴 BLOCKER: `should_index_document()` has an inverted return

`ingestion/indexer.py:389-431`. The final line is:

```python
        return doc_state.qdrant_collection == self.config.collection_name
```

Every guard above it returns `True` meaning *"needs indexing"*. This last line
returns `True` when the collection **matches** — i.e. when the document is
perfectly up to date. It should be `!=`.

**Consequences today:**
- A document that is fully current (same fingerprint, same model, same
  collection) returns `True` → **re-embedded on every single run**.
- A document whose collection has genuinely changed returns `False` → **silently
  skipped**, the one case the check exists for.

The spec (§2, §3.1) explicitly leans on this function: *"already implements
exactly the 're-embed only on change' rule."* It does not. Layer 2 is inverted,
so the spec's acceptance test — *"a second run does no re-embedding"* — cannot
pass on the current code no matter how well Layer 1 is written. **Fix this
first**; it is a one-character change plus a regression test.

### F2 — 🔴 Per-ticker collections are a hard requirement and the spec assumes the opposite

Your instruction: *"Each stock should have their own independent embeddings on
Qdrant vector store."* The spec assumes a **single shared** collection
(`grownxt_financial_elements`, `indexer.py:50`) with a `ticker` payload filter
(`indexer.py:872-878`), and its example manifest hardcodes one
`qdrant_collection` for the whole run.

This is a real architectural change, not a naming tweak, and it is not in the
spec at all. See §C.1 for the design. The good news: `ingestion/rag/retriever.py`
reaches Qdrant **only** through `indexer.search()` / `search_by_vector()` /
`search_batch_by_vectors()`, and every one of those already receives the ticker,
so the change is contained inside `ingestion/indexer.py`.

### F3 — 🔴 Qdrant local `path=` mode is single-process exclusive; it breaks §3.8

Spec §2.2 recommends option (b): `QdrantClient(path=...)` embedded mode. Spec
§3.8 simultaneously requires a **detached background process** running the batch
while `app.py` serves reports and RAG search from the same store.

`QdrantClient(path=...)` takes an **exclusive file lock**. Two processes cannot
open it. The combination the spec proposes cannot work: either the running
server blocks the batch job at startup, or the batch job blocks the server.
The spec never notices this.

Resolution options in §B.1. Recommendation: a Qdrant **server** (a container in
`docker-compose.yml`, or Qdrant Cloud). Embedded `path=` mode stays supported,
but only as a CLI-only / server-not-running dev fallback, and the CLI must fail
with a clear message rather than hanging on the lock.

### F4 — 🟠 `vectors.npz` / `payloads.json` are read but **never written**

`ingestion/rag/retriever.py:306` opens `output/<TICKER>/vectors.npz` as a
"Tier-1 sub-millisecond local matrix search" fast path, and `README.md:65,104,171,177`
documents it as a shipped feature. **Nothing in the codebase writes that file.**
`grep -rn "savez"` over the repo returns nothing outside the README. The Tier-1
path is dead; every retrieval silently falls through to Qdrant.

This matters here for two reasons:
1. It is the most natural expression of *"each stock has its own independent
   embeddings"* — a self-contained per-stock vector file on disk.
2. It makes the embedding work **durable independent of Qdrant**. Even against
   an ephemeral store, `output/<TICKER>/vectors.npz` survives and can rebuild the
   collection with no re-embedding.

Also: that path is hardcoded relative (`Path(f"output/{sym}/vectors.npz")`), so
it ignores `GROWNXT_OUTPUT_DIR` and is wrong whenever CWD ≠ project root.

Writing this cache is folded into the plan (§C.2, Phase 4).

### F5 — 🟠 No logging design at all

The spec's only observability is `run_manifest.json` plus "a structured log
line". You asked specifically for a `logs/` folder, document-fetch counts, and
embedding detail. There is currently **no `logs/` directory, no file handler,
and no rotating handler anywhere in the repo** — `scripts/cli.py:39-54` sets up
console-only `basicConfig`. Fully designed in §D.

### F6 — 🟠 A chunker/extractor version bump will be silently ignored

`ChunkSet.fingerprint` includes `params`, which includes `CHUNKER_VERSION`
(`chunker.py:586-592`), so Layer 2 *would* catch a chunker bump. But Layer 1
skips any `doc_id` already `INDEXED` and therefore **never re-chunks it** — the
old `chunks/<doc_id>.json` on disk keeps its old fingerprint, so Layer 2 sees no
change either. Same for `EXTRACT_VERSION` (`extract.py:64`).

Net effect: bumping the chunker or extractor changes nothing until someone
remembers to run `--force-all`. Fix: record `chunker_version` / `extract_version`
in `state.json` per document and treat a mismatch as "pending" in Layer 1 (§C.3).

### F7 — 🟡 `fetch_catalog` raises on an empty catalogue

`catalog.py:295-296` raises `CatalogError` when a ticker's catalogue lists zero
documents. For a 50-ticker batch this is a routine per-ticker outcome (a newly
added constituent with no filings mirrored yet), not an error. The status
taxonomy needs a distinct `NO_DOCUMENTS` state so it is not reported as a
failure and does not turn the run red.

### F8 — 🟡 Payload duplicates the full chunk text, roughly doubling store size

`build_chunk_payload()` (`indexer.py:548-570`) stores both `content` (raw chunk
text) and `embed_text` (the same text plus a header). At ~1500–3000 chunks per
ticker × 50 tickers, that is 75k–150k points, and the duplicated text dominates
the footprint (rough order: ~450 MB of vectors at 768×float32, plus **~500 MB of
payload text where ~250 MB is a duplicate**). `embed_text` is never read back by
the retriever — `retriever.py:415-424` projects only `content, label, page_start,
page_end, doc_type, section_breadcrumb, chunk_id, ticker`.

Recommend dropping `embed_text` from the payload (or truncating it) as part of
Phase 1. Low risk, meaningful saving at Nifty-50 scale.

### F9 — 🟡 Realistic scale is larger than the spec's estimate

Spec §6 assumes "~5 documents each". With the spec's own defaults
(`annual_reports=1, concall_years=1`), the catalogue returns 1 annual report +
up to 4 transcripts + up to 4 presentations ≈ **5–9 documents per ticker**, so
**250–450 documents** for the full index, not 250. Cold-run estimate should be
stated as **8–15 hours sequential**, dominated by Docling on annual reports
(transcripts hit the fast text path, `extract.py:234`).

### F10 — 🟡 Minor: `DocumentStore` has no `chunks` property

`storage.py` owns the path layout and its docstring even names `chunks/`
(line 14), but there is no `chunks` property and `ensure()` (line 96) does not
create it. Two call sites hardcode the string instead —
`indexer.py:821` and `pipeline.py:209`. Add the property; it is the module that
is supposed to own this.

---

## B. Decisions needed before Phase 1

I have a recommendation for each; the plan below is written assuming the
recommended option, and each is isolated enough to swap.

### B.1 — Qdrant deployment target (blocks Phase 1)

| Option | Pros | Cons |
| :--- | :--- | :--- |
| **(Rec.) Qdrant server container** — add a `qdrant` service to `docker-compose.yml`, `QDRANT_URL=http://qdrant:6333` | Multi-process safe (fixes F3); real persistence; free; matches the batch+server topology the spec needs | One more container locally; needs a host port for bare-metal dev |
| Qdrant Cloud free tier | Zero local infra; survives HF Space restarts | Needs an account + `QDRANT_API_KEY`; free tier storage is tight at F8's footprint unless `embed_text` is dropped |
| Embedded `QdrantClient(path=...)` | No service at all | ❌ Single-process lock — cannot satisfy spec §3.8 |

**Recommendation**: server container as the default, `QDRANT_URL`/`QDRANT_API_KEY`
for Cloud, embedded `QDRANT_LOCAL_PATH` supported but documented as CLI-only.
Combined with F4's `vectors.npz`, the embeddings survive regardless of which is
chosen.

### B.2 — Source of the Nifty 50 constituent list (blocks Phase 0)

The spec is right to refuse a memorised list: today is **2026-09-03**, and NSE
runs semi-annual reviews effective end-March and end-September, so at least two
rebalances have happened since my training data. Options: **(a)** you paste the
current list, **(b)** I fetch NSE's public constituent CSV at implementation
time. **Recommendation: (a)** — NSE actively blocks non-browser clients, so a
fetch is likely to need cookie priming and may silently return stale or partial
data.

### B.3 — Backfill depth

Spec default is `annual_reports=1, concall_years=1` (≈5–9 docs/ticker).
**Recommendation**: keep 1/1 for the first full backfill to get complete Nifty-50
coverage in one overnight run, then deepen selectively with
`--annual-reports 3 --tickers ...`. Deeper on the first pass roughly triples an
already 8–15h run.

### B.4 — Notification channels and API auth

Spec §3.9 offers log/manifest (always on), webhook, and email; §7 Q5 asks whether
`POST /api/embeddings/nifty50/run` should be authenticated.
**Recommendation**: ship log/manifest + optional webhook only (email is extra
credential surface for no added signal), and gate the POST route behind a
`NIFTY50_ADMIN_TOKEN` bearer check — triggering 8–15 hours of CPU from an
unauthenticated public Space endpoint is a denial-of-service handle.

---

## C. Target design (deltas from the spec)

### C.1 — Per-ticker Qdrant collections (F2)

Contained entirely within `ingestion/indexer.py`.

```python
# IndexerConfig
collection_prefix: str = "grownxt"  # QDRANT_COLLECTION_PREFIX
collection_per_ticker: bool = True  # QDRANT_COLLECTION_PER_TICKER
collection_name: str = DEFAULT_COLLECTION_NAME  # shared-mode fallback, unchanged


def collection_for(self, ticker: str) -> str:
    """Returns the collection a ticker's vectors live in."""
    if not self.collection_per_ticker:
        return self.collection_name
    return f"{self.collection_prefix}_{safe_ticker(ticker)}".lower()
```

- `safe_ticker()` already yields only `[A-Z0-9_-]`, which is a valid Qdrant
  collection name — `M&M` → `grownxt_m_m`. No extra sanitisation needed.
- Every method that currently reads `self.config.collection_name` takes a
  `ticker` (or resolves it from the `ChunkSet`) and calls `collection_for()`:
  `init_collection`, `_create_payload_indexes`, `upsert_points`,
  `index_chunk_set`, `search_by_vector`, `search_batch_by_vectors`, `search`.
- `search_batch_by_vectors` currently batches into one collection; every caller
  (`retriever.py:426`) already batches **within a single ticker**, so it takes
  one `ticker` argument. Assert that all requests in a batch share a ticker.
- **Migration is free**: `state.json` already records `qdrant_collection` per
  document (`indexer.py:749`), and `should_index_document` compares it — so once
  F1's inverted return is fixed, flipping to per-ticker collections
  automatically marks every document as needing re-index, with no manual purge.
- **Trade-off to accept**: 50 collections each carry their own HNSW graph, so
  idle RAM is higher than one shared collection. At this corpus size that is
  acceptable, and the isolation buys per-stock drop/rebuild
  (`DELETE /collections/grownxt_tcs`) and removes the ticker filter from the hot
  search path.
- Keep `collection_per_ticker=False` working so the shared-collection mode
  remains available.

### C.2 — Per-stock local vector cache (F4)

After a ticker's documents are indexed, write the two files the retriever
already expects, under `OUTPUT_DIR/<TICKER>/`:

- `vectors.npz` — `np.savez_compressed(vectors=<float32 (n, 768)>, chunk_ids=<str>)`
- `payloads.json` — the projected payload fields, aligned by row index

Written atomically (temp + `os.replace`), regenerated whenever any of that
ticker's documents were re-indexed. This is what makes each stock's embeddings
genuinely independent and portable, and it lights up the retriever's Tier-1 path
for the first time. Also fix `retriever.py:306` to resolve through
`indexer.config.output_dir` instead of the hardcoded relative `output/`.

### C.3 — Layer 1 pending-set rule (extends spec §3.4, fixes F6)

A catalogue entry is **pending** when any of:

1. `doc_id` is absent from `state.json`, **or**
2. its recorded `status != "INDEXED"` (retry a prior failure — spec's own note), **or**
3. its recorded `chunker_version != CHUNKER_VERSION` or
   `extract_version != EXTRACT_VERSION` (**new**, fixes F6), **or**
4. its recorded `qdrant_collection != collection_for(ticker)` (**new**, catches the
   per-ticker migration), **or**
5. the ticker is in `--force` / `--force-all`.

This requires adding `chunker_version` and `extract_version` to
`DocumentIndexState` (both defaulting to `""`, so old state files load and are
treated as pending — a one-time re-index, which is correct after F1's fix
anyway).

### C.4 — Run status taxonomy (fixes F7)

Per-ticker: `OK` | `PARTIAL` | `FAILED` | `NO_DOCUMENTS` | `SKIPPED`.
Run-level: `RUNNING` | `COMPLETED` | `COMPLETED_WITH_ERRORS` | `FAILED`.
`NO_DOCUMENTS` counts toward "completed", not toward the failure exit code.

---

## D. Logging & observability design (fills F5)

Two independent surfaces, both always on, neither requiring configuration:

### D.1 — Layout

```
logs/                                  # NEW, gitignored; GROWNXT_LOG_DIR overrides
└── nifty50/
    ├── nifty50_embedding.log          # rolling human-readable, all runs (10 MB × 5)
    ├── latest.json                    # {"run_id": ..., "log_dir": ..., "status": ...}
    └── runs/<run_id>/
        ├── run.log                    # full DEBUG log for this run only
        ├── events.jsonl               # one JSON object per line, machine-readable
        └── summary.json               # final RunSummary (same shape as the manifest entry)
```

`<run_id>` is `YYYYMMDDTHHMMSSZ`. `--background` also redirects the child
process's raw stdout/stderr into `runs/<run_id>/run.log`, so a hard crash inside
Docling still leaves a traceback on disk.

New module: **`ingestion/runlog.py`** — owns handler setup and the event writer,
so `scripts/cli.py` stays console-only and unchanged for every other script.

```python
def setup_run_logging(run_id: str, log_dir: Path | None = None,
                      verbose: bool = False) -> RunLogger
```

Attaches a `RotatingFileHandler` (rolling log, `TIMED` format from
`scripts/cli.py:19`) and a per-run `FileHandler` at DEBUG to the **root** logger,
so records from `ingestion.catalog`, `ingestion.documents.download`,
`ingestion.documents.extract`, `ingestion.chunker`, and `ingestion.indexer` are
captured automatically without touching those modules. All handlers open with
`encoding="utf-8"` — filings carry `₹` and typographic dashes, and this is the
exact crash `console_utf8()` exists to prevent (`CLAUDE.md`: Console & Encoding).

### D.2 — Structured event stream (`events.jsonl`)

Every event: `{"ts", "run_id", "event", "ticker", ...fields}`. This is the
"log all information" surface — it is what makes "how many documents were
fetched" answerable with one `jq` line rather than by reading prose.

| Event | Fields |
| :--- | :--- |
| `run_started` | `tickers_in_scope`, `mode` (full/partial/dry-run), `force`, `annual_reports`, `concall_years`, `workers`, `embedding_model`, `vector_size`, `qdrant_target`, `collection_per_ticker`, `nifty50_as_of`, `pid`, `python`, `platform` |
| `ticker_started` | `index` (n of 50), `symbol`, `company_name` |
| `catalog_fetched` | `entries_total`, `by_doc_type` {annual_report, concall_transcript, concall_presentation}, `elapsed_s`, `screener_url` |
| `diff_computed` | `known`, `pending`, `pending_doc_ids`, `reason_counts` {new, retry_failed, version_bump, collection_change, forced} |
| `download_completed` | `doc_id`, `ok`, `reused` (cache hit vs fetched), `bytes`, `http_attempts`, `elapsed_s`, `error` |
| `extract_completed` | `doc_id`, `pages`, `blocks`, `tables`, `figures`, `fast_path` (transcript shortcut), `elapsed_s` |
| `chunk_completed` | `doc_id`, `n_chunks`, `n_chars`, `element_counts` {text, table, figure}, `fingerprint`, `chunker_version` |
| `embed_completed` | `doc_id`, `chunks_embedded`, `model`, `vector_dim`, `batches`, `elapsed_s`, `chunks_per_s` |
| `qdrant_upsert` | `doc_id`, `collection`, `points_upserted`, `collection_count_after` (verified by a real `client.count()`), `elapsed_s` |
| `local_cache_written` | `path`, `vectors`, `bytes` |
| `ticker_completed` | `status`, `new_documents`, `unchanged_documents`, `failed_documents`, `chunks_indexed`, `points_total`, `elapsed_s`, `errors[]` |
| `run_completed` | `status`, totals for every counter below, `elapsed_s` |
| `notify_sent` / `notify_failed` | `channel`, `status_code`, `error` |

**Run-level counters carried in `run_completed` and `summary.json`** — the
headline numbers you asked to be logged correctly:

```
tickers_total / ok / partial / failed / no_documents
documents_catalogued        documents_pending
documents_downloaded        documents_reused_from_cache      documents_download_failed
documents_extracted         documents_extract_failed
documents_chunked           chunks_total
documents_embedded          chunks_embedded                  vectors_upserted
bytes_downloaded            collections_touched
elapsed_seconds             elapsed_by_stage {download, extract, chunk, embed, upsert}
```

### D.3 — Console

Unchanged in style from `scripts/ingest_documents.py`: a per-ticker
`doc_id | status | chunks | model` grid, a closing 50-row summary table, and the
final `"N/50 tickers OK, M partial, K failed, J no documents."` line. Exit code
non-zero iff any ticker is `FAILED` or `PARTIAL`.

---

## E. Implementation phases

Ordered so each phase is independently verifiable and the risky/blocking work
lands first. Phases 1–2 are worth doing even if the batch script is deferred.

### Phase 0 — Data & branch (½ hour)
- Cut `feat/nifty50-embeddings`.
- Resolve **B.2**; write `ingestion/data/nifty50.json` (`index`, `as_of`,
  `source`, `constituents[{symbol, name}]`) exactly as spec §3.3 shapes it.
- `ingestion/data/__init__.py::load_nifty50(path: Path | None = None) -> list[str]`.
- **Accept**: 50 unique symbols; no two collide under `safe_ticker()`.

### Phase 1 — Fix the indexer (blocking; ~half a day)
Files: `ingestion/indexer.py`, `ingestion/documents/storage.py`

1. **F1**: `should_index_document` → `return doc_state.qdrant_collection != self.config.collection_for(ticker)`.
2. **F2/C.1**: add `collection_prefix`, `collection_per_ticker`, `collection_for()`;
   thread `ticker` through `init_collection`, `_create_payload_indexes`,
   `upsert_points`, `index_chunk_set`, `search*`.
3. **F6/C.3**: add `chunker_version` + `extract_version` to `DocumentIndexState`
   (and populate them in `index_chunk_set` from `chunk_set.params`).
4. **F3/B.1**: `IndexerConfig.from_env()` reads `QDRANT_LOCAL_PATH`; client
   precedence becomes URL → local path → `:memory:`, and the `:memory:` fallback
   logs at **ERROR** ("vectors will be lost on exit") rather than WARNING.
5. **F8**: drop `embed_text` from `build_chunk_payload` (keep the local variable —
   it is still what gets embedded).
6. **F10**: add `DocumentStore.chunks` property + include it in `ensure()`;
   replace the two hardcoded `"chunks"` strings.
7. Resolve **B.1** infra: `qdrant` service in `docker-compose.yml`, `QDRANT_*`
   rows in `.env.example`.

**Accept**: index one ticker twice; second run reports every document `SKIPPED`
and issues zero `encode()` calls. *(This is the check that fails today.)*

### Phase 2 — Logging spine (~half a day)
Files: **new** `ingestion/runlog.py`, `.gitignore`

- `setup_run_logging()`, `RunLogger.event(name, **fields)` (append-only JSONL,
  flushed per line so a `tail -f` is live), `RunLogger.counters`, atomic
  `summary.json` write, `latest.json` pointer.
- `logs/` added to `.gitignore`.
- **Accept**: a synthetic run produces all four files; `events.jsonl` is valid
  JSONL; UTF-8 content (`₹`) round-trips on Windows.

### Phase 3 — `run_batch()` core (~1 day)
File: **new** `scripts/embed_nifty50.py`

- `run_batch(tickers, *, force, annual_reports, concall_years, workers, dry_run,
  run_id, log_dir, ...) -> RunSummary` — one plain function, exactly as spec
  §3.8 requires, called by every entry point.
- Per-ticker algorithm from spec §3.4 with the C.3 pending rule and C.4 status
  taxonomy; `CatalogError` → `NO_DOCUMENTS`, never a crash.
- `RunManifest` dataclass ↔ `output/_nifty50/run_manifest.json`, written
  atomically after **every ticker** (so a poll mid-run sees real progress), with
  `to_dict`/`from_dict` and corrupt-file tolerance mirroring
  `load_ticker_state` (`indexer.py:364-369`).
- Emit every event in §D.2 at its stage boundary.
- 1–2 s delay between tickers' catalogue calls (spec §3.5).
- `--workers` capped at 4.

**Accept**: `--dry-run --limit 3` fetches 3 catalogues, writes a manifest and a
full event stream, downloads nothing.

### Phase 4 — Per-stock local vector cache (~2 hours)
Files: `ingestion/indexer.py`, `ingestion/rag/retriever.py`

- `write_local_vector_cache(ticker)` → `vectors.npz` + `payloads.json` (C.2),
  called by `run_batch` after a ticker's indexing when anything changed.
- Fix `retriever.py:306` to resolve via `indexer.config.output_dir`.
- **Accept**: after one ticker, both files exist and the retriever logs its
  Tier-1 matrix path instead of falling through to Qdrant.

### Phase 5 — CLI surface (~half a day)
File: `scripts/embed_nifty50.py`

- Every flag in spec §3.4, plus `--tail`.
- `--background`: `subprocess.Popen([sys.executable, "-m", "scripts.embed_nifty50", ...])`
  with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows /
  `start_new_session=True` on POSIX, stdout+stderr → `runs/<run_id>/run.log`;
  writes `status: RUNNING` + `pid`, prints `run_id` and log path, returns.
- `--status [--run-id ID] [--json]`: pure file read, zero network.
- Single-flight: refuse to start when the manifest says `RUNNING` **and** the pid
  is alive (`os.kill(pid, 0)` / `OpenProcess` on Windows), with a staleness
  fallback of 24 h — spec §6 leaves the timeout unspecified; pin it.
- `console_utf8()` at entry (`CLAUDE.md` requirement).

**Accept**: `--background --limit 1` returns in < 2 s printing a `run_id`;
`--status` then shows `RUNNING`, and later a terminal status.

### Phase 6 — API + notifier (~half a day)
Files: `app.py`, **new** `ingestion/notify.py`

- `POST /api/embeddings/nifty50/run` → validates, launches the **same** detached
  subprocess as Phase 5, returns `202` + `{run_id, status_url}`; `409` when a run
  is live; bearer check against `NIFTY50_ADMIN_TOKEN` (B.4) → `401`/`503` when
  unset.
- `GET /api/embeddings/nifty50/status[/{run_id}]` → manifest read, includes
  `completed_tickers/total_tickers`.
- Notifier fires **exactly once**, after the last ticker's
  `index_ticker_documents()` returns (Qdrant already acked via `wait=True`,
  `indexer.py:633-637`) **and** a verification `client.count()` has run per
  touched collection. Log/manifest always on; webhook optional
  (`NIFTY50_NOTIFY_WEBHOOK_URL`, best-effort, one retry, never raises);
  `notified_at` stamped in the manifest so a restarted poller does not double-send.

### Phase 7 — Verification + docs (~half a day)
Files: **new** `scripts/verify_nifty50_embeddings.py`; `CLAUDE.md`, `.env.example`, `README.md`

See §F. Docs: new script in **Development Commands**, the two routes in
**REST Endpoints**, and `QDRANT_URL`/`QDRANT_API_KEY`/`QDRANT_LOCAL_PATH`/
`QDRANT_COLLECTION_PREFIX`/`QDRANT_COLLECTION_PER_TICKER`/`GROWNXT_LOG_DIR`/
`NIFTY50_NOTIFY_WEBHOOK_URL`/`NIFTY50_ADMIN_TOKEN` in **Configuration**. Correct
the README's `vectors.npz` claim from aspiration to fact (it becomes true in
Phase 4). Note in `CLAUDE.md` that `scripts/ingest_documents.py` and
`scripts/verify_ingestion.py` are broken (F-list §2.1 of the spec) — documenting
dead commands as live ones is its own bug.

---

## F. Verification plan (`scripts/verify_nifty50_embeddings.py`)

Built on the `Report`/`Check`/`banner`/`require` harness in `scripts/checks.py`
(as `verify_reporting.py` does — **not** on the broken `verify_ingestion.py`).

**Offline (no network, CI-safe):**
1. `nifty50.json` parses; exactly 50 unique symbols; no `safe_ticker()` collision.
2. **F1 regression**: a `TickerState` recording an up-to-date document →
   `should_index_document()` returns `False`. *(Fails on today's code — this is
   the guard that keeps the bug from coming back.)*
3. Layer-1 pending set: synthetic `Catalog` + `TickerState` covering all five C.3
   reasons returns exactly the expected `doc_id`s.
4. `collection_for()`: per-ticker on/off, `M&M` → `grownxt_m_m`, prefix override.
5. Manifest `to_dict`/`from_dict` round-trip; a corrupt/missing manifest degrades
   to "no history" instead of raising.
6. `--status` against 2–3 fake ticker directories makes **zero** network calls
   (stub `fetch_catalog` and assert it was never called).
7. `--background` returns in < 2 s and prints a `run_id` (stub `Popen`; assert it
   was called once with `--background` stripped).
8. Notifier fires **once** per run, only after `status` leaves `RUNNING`; a
   webhook that raises is caught, logged, and does not change the recorded status.
9. **Logging**: a synthetic run writes all four files; `events.jsonl` is valid
   JSONL; the counters in `run_completed` equal the sum of the per-document
   events; `₹` round-trips through the file handlers.

**Live (opt-in, `--live TCS`):**
10. First run → ≥1 document `INDEXED`; counters match the events.
11. Immediate re-run → `new_documents: 0`, **zero** downloads, **zero** Docling
    calls, **zero** `encode()` calls, `status: OK`. *(Spec §1's acceptance
    criterion.)*
12. **Persistence**: a fresh Python process reconnects and
    `client.count(collection_for("TCS"))` returns the same non-zero total — the
    check that would have caught the in-memory-Qdrant problem.
13. **Isolation**: `grownxt_tcs` exists and contains only TCS points; deleting it
    leaves other tickers' collections intact.
14. `vectors.npz` + `payloads.json` exist, row counts agree with the Qdrant count,
    and the retriever takes its Tier-1 path.
15. `--background` for one ticker: CLI returns before completion; poll
    `--status --run-id ID` until it leaves `RUNNING`; exactly one webhook POST
    arrives, timestamped **after** the Qdrant count increased.

**Rollout** (spec §8, last item): `--limit 3 --dry-run` → real `--limit 3` →
second `--limit 3` (expect a full no-op) → `--limit 3 --background` → full 50.

---

## G. Risks & operational notes

| Risk | Mitigation |
| :--- | :--- |
| **F1 masked as "slow"** — before the fix, every run re-embeds everything, which looks like "embedding is just slow" rather than a bug | Phase 1 first; check #11 makes the regression loud |
| **Ephemeral HF Space filesystem** | `vectors.npz` (C.2) + a server/Cloud Qdrant means a wiped Space costs re-download+re-extract but never re-embedding of surviving artifacts; §3.1's per-`doc_id` state makes an interrupted run resume cleanly |
| **Disk** — 250–450 PDFs + `extracted/` + `figures/` + `chunks/` across 50 tickers | Realistically 15–40 GB under `OUTPUT_DIR`; check free space before the first full run. The Docker volume `grownxt_output` must be sized for it |
| **RAM** — 50 collections' HNSW graphs + the Arctic model | Accepted trade-off of C.1; watch the Qdrant container's RSS during the first full run |
| **Cold-run duration 8–15 h** (F9) | `--limit`-staged rollout; resumable by design; run detached |
| **Financial Data Collector is a shared free service** | 1–2 s inter-ticker delay + the existing retry/backoff (`catalog.py:264-286`) |
| **NSE list goes stale** | `as_of` in `nifty50.json`; verification warns when it is > 6 months old (a rebalance has certainly happened) |
| **Unauthenticated trigger route** | B.4's bearer token; route returns 503 when the token is unset rather than defaulting to open |

---

## H. Checklist

- [ ] **B.1** Qdrant target decided; **B.2** constituent list sourced; **B.3** depth confirmed; **B.4** notify/auth confirmed
- [ ] Phase 0 — `ingestion/data/nifty50.json` + `load_nifty50()`
- [ ] Phase 1 — F1 inverted return, per-ticker collections, version fields, `QDRANT_LOCAL_PATH`, payload slimming, `DocumentStore.chunks`, compose + `.env.example`
- [ ] Phase 2 — `ingestion/runlog.py`, `logs/` layout, JSONL events, counters
- [ ] Phase 3 — `run_batch()`, manifest, per-ticker loop, status taxonomy
- [ ] Phase 4 — `vectors.npz`/`payloads.json` writer; retriever path fix
- [ ] Phase 5 — full CLI, `--background`, `--status`, single-flight
- [ ] Phase 6 — two API routes (authenticated) + one-shot notifier
- [ ] Phase 7 — `scripts/verify_nifty50_embeddings.py`; `CLAUDE.md`/`.env.example`/`README.md` updated
- [ ] Rollout — `--limit 3` dry → real → no-op → background → all 50
- [ ] DoD (`CLAUDE.md`): Google style + type annotations, `ruff` clean, verification suite green, no credentials committed, API contracts preserved
