# Spec: Resumable Nifty 50 Embedding via a Checkpointed LangGraph Orchestrator

- **Status**: Draft, for review
- **Author**: Claude (via /specs), for @jenish.gajera
- **Date**: 2026-09-03
- **Target branch**: new `feat/nifty50-resumable-graph` off `feat/tier2-composites-and-verification`
- **Supersedes**: nothing. **Extends**: `.claude/specs/nifty50-embedding-pipeline.md` (§3.1 two-layer
  change detection, §3.6 run manifest, §3.8 detached execution). Every contract that spec
  established stays intact; this one adds a resume layer beneath it and closes the gaps the
  as-built pipeline has.

---

## 1. Goal

Make an interrupted Nifty 50 embedding run resume **cheaply and at fine granularity**, and make
the resume behaviour *observable* rather than inferred.

Two independent problems, and the spec is explicit that they need two different mechanisms:

1. **"Where was I?"** — today nothing records mid-ticker position. A restart re-visits all 50
   tickers from the top and re-runs every stage of any ticker whose documents were not yet
   embedded. **Solved by**: a LangGraph `StateGraph` with a durable checkpointer (§4.2–4.6).
2. **"How expensive is it to get back there?"** — today the Docling extraction, the single most
   expensive stage in the pipeline, is re-run even though its output is complete on disk.
   **Solved by**: making the artifact caches readable (§4.9). *A checkpointer cannot fix this.*

The distinction is the core design principle of this document:

> **The checkpointer records where you were. The artifact caches make getting back there cheap.
> Shipping only the checkpointer produces a pipeline that resumes precisely to a point it then
> pays full price to re-reach.**

Success criteria:

- A run killed at any point resumes with **zero repeated Docling passes** over documents already
  extracted, and **zero repeated HTTP requests** for documents already downloaded or catalogues
  already checked within the TTL.
- Resuming a run that died at ticker 40 of 50 reaches new work in **seconds**, not minutes.
- A run that cannot be resumed correctly (ephemeral vector store) **refuses to start** rather than
  recording success for vectors that no longer exist.
- `scripts/verify_nifty50_embeddings.py` gains offline kill-and-resume coverage and stays green.

---

## 2. Background — what resumption does today

Audited against the current tree. Line references are to `HEAD` of
`feat/tier2-composites-and-verification`.

### 2.1 There is no run-level checkpoint

`ingestion/batch.py` has no resume concept and `scripts/embed_nifty50.py` has no `--resume` flag.
Resumption is *reconstructed* from durable per-ticker state on every fresh run:

- `run_batch` (`batch.py:1103`) re-enumerates every constituent and calls `process_ticker`
  (`batch.py:672`) for each. `finish_ticker` (`batch.py:1213`) writes
  `output/_nifty50/run_manifest.json` after each ticker, **but nothing ever reads it back to skip
  a ticker**. It is a report, not a resume record.
- The only authoritative "done" marker is `output/<TICKER>/state.json`, written by
  `QdrantVectorIndexer.index_chunk_set` after each successful upsert (`indexer.py:1010-1019`).
- Layer 1 (`pending_reason`, `batch.py:561`) reads that file and re-queues anything not recorded
  as `INDEXED` under the current chunker version, extractor version and collection.

Consequence: **the commit point is the Qdrant upsert.** Everything upstream of it is redone.

### 2.2 `process_ticker` is stage-batched, not per-document pipelined

```
catalogue (batch.py:721)
  -> Layer 1 diff (batch.py:604)
  -> ALL pending PDFs download concurrently, blocking (batch.py:853-863)
  -> per document, sequentially: extract (900) -> chunk (945) -> write chunk file
  -> THEN Layer 2: scan chunks/*.json, embed each (batch.py:980-989)
  -> verify point count, mirror vectors.npz (batch.py:1052)
```

Because nothing is embedded until *every* document of that ticker is chunked, a crash late in a
ticker discards all of that ticker's extraction and chunking work.

### 2.3 What survives, and what is actually re-read

| Artifact | Path | Re-read on restart? |
| :-- | :-- | :-- |
| PDF | `output/<T>/documents/<doc_id>.pdf` | **Yes** — validated and reused (`download.py:371`) |
| Extraction JSON | `output/<T>/extracted/<doc_id>.json` | **No** — written, never read back |
| Chunk JSON | `output/<T>/chunks/<doc_id>.json` | Yes, by Layer 2 — but after Layer 1 already overwrote it |
| `state.json` | `output/<T>/state.json` | **Yes** — authoritative |

`Extractor.run()` (`extract.py:285`) writes its cache and never consults it. `read_extraction()`
exists (`extract.py:886`) and has no caller in the batch path. CLAUDE.md documents this as
intentional — "only Layer 1 prevents a redundant Docling pass" — which is exactly why a crash
*before* the upsert costs a full re-extraction.

### 2.4 Crash-point behaviour, as built

| Crash during | PDF | Extraction | Chunking | Embedding |
| :-- | :-- | :-- | :-- | :-- |
| Download | completed files reused; interrupted file restarts at byte 0 (no HTTP range) | — | — | — |
| Extraction of doc *n* | all reused | **all pending docs re-extracted, incl. completed ones** | re-run | — |
| Chunking | reused | **re-extracted** | re-run | — |
| Embedding of doc *n* | reused | docs 1..n-1 correctly skipped by Layer 1; doc *n* re-extracted | same | doc *n* restarts at chunk 0 |
| After embed, before `vectors.npz` | — | — | — | self-heals (`batch.py:1052`) |

Notes:
- `_stream` writes to a hidden `.<doc_id>-XXXX.part` and only `replace()`s after validating size,
  PDF magic and a readable page tree (`download.py:476-518`), so a half-written `.pdf` is
  impossible. On a hard kill the `.part` files are orphaned and **nothing sweeps them**.
- Point IDs are `deterministic_uuid(ticker, doc_id, chunk_id)`, so a re-upsert overwrites. Duplicate
  vectors for one chunk cannot occur.
- A failed index writes **nothing** to `state.json` — the `FAILED` values at
  `indexer.py:1047/1063/1077` are on the returned `IndexResult`, not on a `DocumentIndexState`.
  So `REASON_RETRY` is effectively unreachable and such a document reads back as `REASON_NEW`.
  Same outcome, misleading label in the logs.
- Transcripts are cheap to redo — `_extract_fast_transcript` (`extract.py:392`) bypasses Docling
  entirely. The loss is concentrated on annual reports and investor presentations.

### 2.5 The single-flight lock will not block a restart

`run_batch` catches `BaseException`, marks the manifest `FAILED` and re-raises inside a `finally`
that saves it, so Ctrl+C leaves a clean non-`RUNNING` manifest. For a hard kill that leaves
`status: RUNNING`, `active_run` (`batch.py:524`) checks `pid_alive` (`batch.py:469`) — a dead pid
means abandoned. Only when no pid was recorded does it fall back to `STALE_RUN_HOURS = 24`
(`batch.py:77`). This is correct and needs no change; this spec only adds the ability to *resume*
such a manifest rather than restart it.

### 2.6 LangGraph is already a dependency

- `langgraph==1.2.11`, `langgraph-checkpoint==4.2.0` are in `requirements.lock`; `langgraph>=0.2.0`
  is in `requirements.txt` and `pyproject.toml`. **No new dependency is required.**
- Already used in `ingestion/rag/pipeline.py`: a 5-node `StateGraph(ResearchState)` over a
  `TypedDict`, `compile()`d **without a checkpointer**, behind an
  `ImportError`-tolerant `LANGGRAPH_AVAILABLE` guard. This spec follows the same
  `TypedDict` + guarded-import conventions.
- **Only `langgraph/checkpoint/{base,memory,serde}` are installed.** There is no
  `SqliteSaver` or `PostgresSaver` — those live in `langgraph-checkpoint-sqlite` /
  `-postgres`, neither of which is in the lockfile. See §4.4.

---

## 3. Practical implications this spec commits to fixing

These are the operational findings from the §2 audit, stated as requirements. **P1–P4 are
independent of LangGraph and deliver the bulk of the wall-clock win** (§8, Phase 1).

**P1 — Restart, do not re-force.** Re-running the identical command is the correct recovery.
Adding `--force` is actively harmful: `pending_reason` returns `REASON_FORCED` for everything and
discards all completed work. *Requirement*: `--resume` must exist as a first-class flag so
operators reach for it instead of `--force`, and `--help` must say so.

**P2 — Do not flip `--fast-tables` across a restart.** `extract_version(accurate_tables)` appends
`+fast-tables`, so flipping it invalidates every document indexed the other way, across all 50
tickers, via `REASON_VERSION`. Device and thread count are correctly excluded from that comparison,
so restarting on a different machine or with different `--threads` is safe. *Requirement*: a resume
must **refuse** to proceed when the recorded run's `fast_tables` differs from the requested one
(§4.14), rather than silently invalidating the corpus.

**P3 — An ephemeral vector store makes resumption silently wrong.** With `QDRANT_API_URL` unset the
store is in-memory; `batch.py:1204-1207` logs an ERROR and carries on. A crash then loses every
vector while `state.json` still says `INDEXED`, so the next run confidently skips everything.
*Requirement*: **abort** instead of warning, unless `--allow-ephemeral` is passed (§4.11).

**P4 — Stop paying for catalogues and extractions already paid for.** Restarting a run that died at
ticker 40 costs 39 redundant catalogue requests plus `--delay` each before reaching new work; and
any ticker mid-flight pays a full Docling re-extraction. *Requirement*: a catalogue TTL (§4.10) and
a cache-reading extractor (§4.9).

**P5 — Narrowing a restart must not require reading JSON by hand.** Today the recipe is
`jq -r '.tickers | to_entries[] | select(.value.status != "OK") | .key' output/_nifty50/run_manifest.json`
diffed against the full constituent list to find tickers never reached. *Requirement*:
`--status` gains a `--pending-only` projection that prints exactly the `--tickers` argument needed
(§4.14).

**P6 — Orphaned `.part` files accumulate.** Harmless (nothing globs for them) but unbounded.
*Requirement*: sweep on ticker entry (§4.12).

**P7 — A failed index should say so in `state.json`.** *Requirement*: record a
`DocumentIndexState(status="FAILED", ...)` on index failure so `REASON_RETRY` becomes reachable and
the logs stop mislabelling retries as new documents (§4.9.3).

---

## 4. Design

### 4.1 Four layers, four granularities

| Layer | Mechanism | Scope | Granularity | Status |
| :-- | :-- | :-- | :-- | :-- |
| **L0** | LangGraph checkpointer, keyed on `thread_id = run_id` | **within one run** | node / superstep | **new** (§4.4) |
| **L1** | catalogue diff vs `state.json` (`pending_reason`) | **across runs** | document | exists |
| **L2** | chunk-set fingerprint (`should_index_document`) | across runs | content | exists |
| **L3** | on-disk artifact caches (PDF, extraction, chunks) | across runs | artifact | PDF only; **extraction + chunks new** (§4.9) |

L0 and L1 are complementary, not redundant, and the spec must not conflate them:

- LangGraph checkpoints are **per `thread_id`**. Resuming the *same* `run_id` gives exact
  node-level resume. A *brand-new* `run_id` gets **no** L0 benefit at all and falls back to L1.
- `state.json` is **cross-thread and cross-run**, and remains the source of truth for
  "is this document embedded". The checkpointer never becomes that authority.

**Rule: if the checkpointer and `state.json` disagree, `state.json` wins.** L0 is an optimisation
for re-entering a run; L2's fingerprint check is the correctness backstop and runs regardless.

### 4.2 Graph topology

New package `ingestion/graph/`. One compiled outer graph; the per-ticker and per-document stages
are subgraphs added as nodes, so they inherit the parent checkpointer and get their own
`checkpoint_ns` automatically.

```
BATCH GRAPH                      thread_id = <run_id>
  START
    -> preflight        # hardware resolution, Qdrant persistence gate (P3), constituent load
    -> dispatch         # emit Send("ticker", ...) for the next batch of un-done tickers
    -> [ticker]  x N    # bounded fan-out, N = resolve_workers(...)
    -> collect          # fold TickerResult into state, write run_manifest.json projection
    -> dispatch         # conditional edge: loop while queue non-empty
    -> finalize         # terminal manifest status
    -> notify           # notify_completion, exactly once (guarded on state.notified_at)
  END

TICKER SUBGRAPH                  checkpoint_ns = ticker:<TICKER>
  START
    -> catalogue        # fetch_catalog, honouring the TTL (§4.10)
    -> diff             # Layer 1: compute_pending
    -> download         # Downloader.fetch_all over pending docs (its own internal retry)
    -> [document] x M   # fan-out per successfully downloaded doc
    -> index            # Layer 2 over chunks/*.json  (RetryPolicy here, §4.7)
    -> mirror           # count_points + write_local_vector_cache
  END

DOCUMENT SUBGRAPH                checkpoint_ns = ticker:<TICKER>|doc:<doc_id>
  START -> extract -> chunk -> END
```

**What this buys over `process_ticker` today.** The `extract` and `chunk` nodes live inside a
per-document subgraph, so a checkpoint lands after *each* document's extraction and after *each*
document's chunking. A crash during the extraction of doc 2 of 3 resumes with doc 1's `extract`
and `chunk` nodes already complete — which, combined with §4.9, means doc 1 costs nothing.

**Ordering change worth calling out.** `index` stays at the ticker level (after all documents),
preserving today's Layer-2 disk-scan semantics and the `_qdrant_target` verification. Moving
`index` into the document subgraph would give an even earlier commit point but changes the
`chunk_ready` / `outcome.status == "SKIPPED" -> "INDEXED"` reconciliation at `batch.py:1030-1035`.
**Deferred to a follow-up**; §6 non-goals.

### 4.3 Graph state schema — and the rule that keeps checkpoints small

`ingestion/graph/state.py`, `TypedDict(total=False)` throughout, matching
`ResearchState` in `ingestion/rag/pipeline.py`.

```
BatchGraphState:
  run_id: str
  options: dict                 # BatchOptions.to_dict()
  hardware: dict                # HardwareProfile.to_dict()
  workers: int
  queue: list[str]              # symbols not yet dispatched
  in_flight: list[str]
  results: Annotated[dict[str, dict], merge_results]   # symbol -> TickerRun.to_dict()
  counters: Annotated[dict[str, int], sum_counters]
  notified_at: str
  status: str

TickerGraphState:
  symbol: str; name: str; collection: str
  catalog_entries: int; selected: list[dict]           # CatalogEntry metadata only
  pending: list[dict]                                  # {doc_id, doc_type, label, reason, source_url}
  unchanged: list[str]
  downloaded: Annotated[list[dict], operator.add]      # {doc_id, relative_path, sha256, n_bytes, reused}
  chunked: Annotated[list[str], operator.add]          # doc_ids with a chunk file on disk
  outcomes: Annotated[dict[str, dict], merge_outcomes]
  errors: Annotated[list[dict], operator.add]

DocumentGraphState:
  symbol: str; doc_id: str; doc_type: str; label: str
  pdf_path: str                 # store-relative
  extraction_path: str          # store-relative
  chunk_path: str               # store-relative
  n_pages: int; n_chunks: int; fingerprint: str
```

> **Hard rule — no payloads in graph state.** `ExtractedDocument`, `ChunkSet`, `Chunk.text`,
> embedding vectors and raw PDF bytes must **never** enter graph state. State carries
> `doc_id`s, store-relative paths, counts and fingerprints; nodes re-open the artifact from disk.
>
> Rationale: a checkpoint is written on **every** superstep. A state holding a 900-chunk `ChunkSet`
> writes megabytes per node, per ticker — turning the checkpointer from an optimisation into the
> new bottleneck, and inflating `output/_nifty50/checkpoints/` without bound. §7 test 4 enforces
> this with a size assertion rather than trusting review.

Reducers are plain named functions (`merge_results`, `sum_counters`, `merge_outcomes`) in
`state.py`, not lambdas — they must be importable for the concurrent `Send` writes to fold
deterministically.

### 4.4 The checkpointer: `JsonFileSaver`

**Constraint.** Only `langgraph/checkpoint/{base,memory,serde}` are installed (§2.6).
`InMemorySaver` dies with the process, which is precisely the failure mode being fixed.
`langgraph-checkpoint-sqlite` would work but CLAUDE.md is explicit — *"No Database Engine: project
is entirely file-backed... Do not introduce SQL or ORM dependencies"* — and the whole platform's
state (`state.json`, `run_manifest.json`, `vectors.npz`) is already file-backed.

**Decision: implement `ingestion/graph/checkpoint.py::JsonFileSaver(BaseCheckpointSaver)`.**

Layout, under `output/_nifty50/checkpoints/`:

```
checkpoints/
  <thread_id>/
    <checkpoint_ns>/                 # "" for the root graph; sanitised for subgraph namespaces
      <checkpoint_id>.json           # {checkpoint, metadata, parent_checkpoint_id, channel_versions}
      <checkpoint_id>.writes.jsonl   # append-only: one record per put_writes call
      latest.json                    # {"checkpoint_id": ...}, replaced atomically
```

Required overrides (these are the four that raise `NotImplementedError` in
`BaseCheckpointSaver`):

- `get_tuple(config)` — resolve `checkpoint_id` from config, else read `latest.json`. Returns a
  `CheckpointTuple` with `pending_writes` loaded from the `.writes.jsonl` sidecar.
- `list(config, *, filter, before, limit)` — newest-first walk of the parent chain, honouring all
  three arguments. `filter` matches against `metadata`.
- `put(config, checkpoint, metadata, new_versions)` — serialise via `self.serde`
  (`JsonPlusSerializer`, inherited), write `<id>.json` then replace `latest.json`. Returns the
  updated `RunnableConfig`.
- `put_writes(config, writes, task_id, task_path="")` — append to the sidecar. Must be idempotent
  per `(task_id, idx)` so a replayed superstep does not duplicate writes.

Also override `delete_thread(thread_id)` (recursive rmtree of the thread directory) so
`--forget-run` (§4.14) works.

Inherited unchanged: `get`, `config_specs`, `get_next_version`, `prune`, `delete_for_runs`,
`copy_thread`, and `get_delta_channel_history` — the last has a working default implementation
against the public contract and is flagged beta upstream; **do not override it.**

Async variants (`aget_tuple`, `alist`, `aput`, `aput_writes`, `adelete_thread`) delegate to the
sync ones via `asyncio.to_thread`. The batch is synchronous, but `app.py` mounts an ASGI server and
a future async caller must not silently block the loop.

**Atomicity**: every write is temp-file + `Path.replace()`, mirroring
`save_ticker_state` (`indexer.py:492-508`) — same helper, same failure logging.

**Thread safety**: `Send` fan-out runs sync nodes on a thread pool, so `put`/`put_writes` are
called concurrently from multiple threads. A single `threading.Lock` around the write path is
sufficient and cheap relative to a Docling pass. Per-thread-id locks are a premature optimisation
here; note it and move on.

**Serialisation guarantee**: `JsonPlusSerializer` handles the `TypedDict`/`dict`/`list`/`str`/`int`
shapes in §4.3. This is only true while the §4.3 hard rule holds — another reason it is tested.

### 4.5 Durability

`graph.invoke(..., durability="sync")`. `Durability = Literal["sync", "async", "exit"]` and the
default is `"async"`, which can lose the final superstep on a hard kill — exactly the scenario
being designed for. `"sync"` writes the checkpoint before the next node starts.

Cost: one small JSON write per superstep, against extraction passes measured in minutes. Not a
tradeoff worth taking. `"exit"` is strictly wrong here and must not be used.

`checkpoint_during` is deprecated in 1.x and must not appear in the implementation.

### 4.6 Bounded fan-out

`resolve_workers` (`batch.py:628`) caps concurrency to 1 under `GPU_CONCURRENCY_VRAM_GB` because
workers are threads sharing one process and each loads its own Docling layout and table models onto
the same card. **LangGraph's `Send` fan-out has no worker-count knob** — every `Send` emitted in a
superstep becomes a task in that superstep. Fanning out 50 tickers at once would reintroduce exactly
the OOM `resolve_workers` was written to prevent.

**Decision: batched dispatch.** The `dispatch` node pops `min(workers, len(queue))` symbols from
`state["queue"]` and returns that many `Send("ticker", ...)` values; a conditional edge from
`collect` routes back to `dispatch` while the queue is non-empty, and to `finalize` when it is
empty. Properties:

- concurrency is bounded by `resolve_workers(options.workers, hardware)` — the existing decision and
  its existing warning log are reused verbatim, not reimplemented;
- a checkpoint lands at **every batch boundary**, so resume granularity is at worst `workers`
  tickers rather than 50;
- dispatch order is deterministic (constituent CSV order), so a resumed run is reproducible.

Rejected alternative: a `threading.BoundedSemaphore` inside the expensive nodes. It works but
parks pool threads, hides the concurrency decision from the graph, and makes the checkpointed state
non-reproducible. Note it in the code comment; do not implement it.

`options.delay_seconds` (inter-ticker politeness) moves into the `catalogue` node, applied when
`state["symbol"]` is not the first of its batch — preserving today's behaviour at `batch.py:1227`.

### 4.7 Retry policy

`RetryPolicy(initial_interval, backoff_factor, max_interval, max_attempts, jitter, retry_on)`.
Apply it **narrowly**:

- `index` node: `RetryPolicy(max_attempts=3, retry_on=<Qdrant transport errors>)`. Qdrant Cloud
  returns transient 5xx/timeouts and `index_chunk_set` currently swallows them into a `FAILED`
  `IndexResult` with no retry at all.
- `catalogue` node: `RetryPolicy(max_attempts=2)` — Screener occasionally 502s.
- `download` node: **no `RetryPolicy`.** `Downloader.fetch` already retries with `_backoff` and
  `Retry-After` handling (`download.py:266-303`) and distinguishes `_Transient` from `_Permanent`.
  A graph-level retry on top would multiply attempts against an exchange server — the opposite of
  the politeness the downloader was written for. Call this out in a comment; it is the kind of
  thing a later contributor "fixes" by adding one.
- `extract` / `chunk` nodes: **no `RetryPolicy`.** A Docling failure on a malformed PDF is
  deterministic; retrying it three times costs three full passes to reach the same error. These
  stay recorded against the document, as today.

### 4.8 Rejected: LangGraph node-level `CachePolicy`

`add_node(..., cache_policy=CachePolicy(key_func=..., ttl=...))` with `compile(cache=...)` looks
like the obvious fit and is the wrong tool here:

- only `langgraph/cache/{base,memory,redis}` are installed. `InMemoryCache` does not survive the
  process — no help for a crash. Redis is a new external service, against the file-backed
  philosophy.
- the default `key_func` hashes the node's **input** with pickle. Our real cache key is the
  *content* — `sha256` of the PDF plus `EXTRACT_VERSION` plus the table mode — which is what L3
  already keys on and what makes the cache correct across machines and runs.
- an artifact cache under `output/<TICKER>/` is inspectable, sweepable and shared with the report
  generator. An opaque cache blob is not.

**Decision: no `CachePolicy` anywhere in this graph.** L3 (§4.9) is the cache. Record the reasoning
in `ingestion/graph/build.py` so it is not relitigated.

### 4.9 L3 — make the artifact caches readable

This is the change that actually removes the wall-clock cost, and it is independent of LangGraph.

#### 4.9.1 Extraction cache

`Extractor.run()` gains `reuse: bool = True`. Before touching Docling (and before the transcript
fast path), when `reuse` is set:

1. `read_extraction(store.extraction(doc_id))` (`extract.py:886`) — `None` means no cache, proceed.
2. Reuse **only if all** of these match:
   - `document.extractor == self.version` — covers `EXTRACT_VERSION` **and** the `+fast-tables`
     suffix, so P2's semantics are preserved exactly;
   - `document.meta["source_sha256"] == <sha256 of the PDF on disk>`;
   - `document.meta["source_bytes"] == <size of the PDF on disk>`;
   - `document.meta["max_pages"] == (int(limit) if limit else 0)`;
   - `document.meta["ocr"] / ["figures"] / ["accurate_tables"] / ["cell_matching"]` match this
     instance;
   - `document.blocks` is non-empty (an empty extraction is a recorded failure, not a cache hit).
3. On reuse, log at INFO with the same shape `_report` uses, emit a `reused=True` flag, and return.
4. On any mismatch, extract normally and overwrite.

**Required prerequisite**: `ExtractedDocument.meta` does not carry the source hash today
(`extract.py:373-384` records `ocr`, `figures`, `accurate_tables`, `cell_matching`, `image_scale`,
`max_pages`, `device`, `num_threads`, `seconds`). Add `source_sha256` and `source_bytes` at write
time. `DownloadResult.sha256` is already computed on the way past in `_stream`
(`download.py:476-518`), so the hash is free on a fresh download and costs one `_hash_file` pass on
a reused PDF — which `_reuse` already performs (`download.py:399`).

`device` and `num_threads` stay in `meta` for observability and are **excluded from the reuse
comparison** — they do not change the extracted text, and including them would invalidate every
cache on every machine change. This mirrors the existing extractor-version contract that CLAUDE.md
already calls out.

**The `write=False` caller stays honest**: `scripts/verify_chunker.py` and friends pass
`write=False`; they must also pass `reuse=False` so the harness always exercises a real Docling
pass.

#### 4.9.2 Chunk cache

The `chunk` node reuses `output/<T>/chunks/<doc_id>.json` when `read_chunk_cache` returns a set
whose `params["version"] == CHUNKER_VERSION`, `params["extractor"]` equals the extraction actually
in hand, and `chunk_size` / `chunk_overlap` match this run's options. Cheap relative to extraction,
but it makes the document subgraph a genuine no-op on a replayed superstep, which is what keeps
§4.5's `durability="sync"` honest.

#### 4.9.3 Record index failures (P7)

`index_chunk_set`'s `except` branch (`indexer.py:1040+`) writes
`DocumentIndexState(status="FAILED", fingerprint=<computed>, chunk_count=len(chunks),
chunker_version=..., extract_version=..., indexed_at=utc_now())` before returning the `FAILED`
`IndexResult`, then `save_ticker_state`.

Effect: `pending_reason` returns `REASON_RETRY` rather than `REASON_NEW`, `diff_computed`'s
`reason_counts` become truthful, and — because `should_index_document` already returns `True` for
any `status != "INDEXED"` (`indexer.py:539-540`) — **no behavioural change to what gets embedded.**
This is a labelling and observability fix, and §7 test 8 pins it so it cannot drift into a skip.

### 4.10 Catalogue TTL (P4)

`TickerState` gains `catalog_checked_at: str` (ISO-8601 UTC), written by the `catalogue` node after
every successful `fetch_catalog`. `BatchOptions` gains `catalog_ttl_hours: float = 6.0`, surfaced as
`--catalog-ttl` and `GROWNXT_CATALOG_TTL_HOURS`.

The `catalogue` node skips the HTTP request and reconstructs `selected` from `state.json` when
**all** hold: the recorded timestamp is inside the TTL; the ticker is not forced; the run is not a
`--dry-run`; and every document `state.json` records for the current
`(chunker_version, extract_version, collection)` triple is `INDEXED`. Any pending or failed document
forces a real catalogue fetch — a stale catalogue must never be the reason work is skipped.

`--catalog-ttl 0` disables it. The skip is logged and emitted as a `catalog_skipped` event
(`ingestion/runlog.py`) with the recorded age, so `--status` and the `events.jsonl` audit trail stay
complete.

Effect on P4: a restart of a run that died at ticker 40 makes **zero** catalogue requests for the 39
completed tickers. Combined with `--resume` (L0), it makes zero requests *and* skips the state reads.
The TTL is the fallback for a fresh `run_id`, where L0 gives nothing.

### 4.11 Refuse an ephemeral vector store (P3)

The `preflight` node replaces `batch.py:1204-1207`. When `not indexer.is_persistent and not
options.dry_run and not options.allow_ephemeral`, it raises `BatchError` with a message naming
`QDRANT_API_URL`, and the run ends `FAILED` **before** any download.

`--allow-ephemeral` / `allow_ephemeral: bool = False` preserves the current behaviour for smoke
tests, downgraded to the existing ERROR log. This is the only spec item that changes an existing
exit code, and it is deliberate: recording `INDEXED` for vectors that evaporated is the one failure
mode where resumption is *silently wrong* rather than merely slow.

### 4.12 Sweep orphaned `.part` files (P6)

`DocumentStore.ensure()` (`storage.py:100`) removes `documents/.*-*.part` older than
`PART_FILE_TTL_HOURS = 6`. The age gate matters: `ensure()` is called from
`process_ticker`/`download` while a concurrent worker may be mid-transfer on the same ticker. Log
each removal at DEBUG with the reclaimed byte count.

### 4.13 `run_manifest.json` stays a projection

The `collect` node writes `output/_nifty50/run_manifest.json` from `BatchGraphState` using the
existing `RunManifest.to_dict()` shape, after every batch of tickers.

**The manifest schema, `GET /api/embeddings/nifty50/status[/{run_id}]`, and
`scripts/embed_nifty50.py --status` do not change.** Status stays a pure file read with no
LangGraph import and no checkpoint parsing — a status poll must not depend on the graph runtime, and
CLAUDE.md's contract requirement (Definition of Done §5) is preserved.

Two additive, optional fields: `resumable: bool` and `resumed_from: str | None`.

### 4.14 CLI surface

New on `scripts/embed_nifty50.py`, additive:

| Flag | Behaviour |
| :-- | :-- |
| `--resume [RUN_ID]` | Resume a run's graph. No `RUN_ID` resumes the newest non-`COMPLETED` manifest. Implemented as `graph.invoke(None, {"configurable": {"thread_id": run_id}}, durability="sync")` — `None` as input is LangGraph's resume-from-checkpoint form. |
| `--no-checkpoint` | Compile with `checkpointer=False`. Reproduces today's behaviour exactly; the escape hatch if the saver misbehaves in production. |
| `--catalog-ttl HOURS` | §4.10. `0` disables. |
| `--allow-ephemeral` | §4.11. |
| `--status --pending-only` | Prints only the symbols not `OK`, space-separated, ready to paste after `--tickers` (P5). With `--json`, an array. |
| `--forget-run RUN_ID` | `checkpointer.delete_thread(run_id)`. Never touches `state.json` or `vectors.npz`. |

**Resume guards.** `--resume` compares the stored `manifest.options` against the invocation and
**refuses** on a mismatch in `fast_tables` (P2), `annual_reports`, `transcripts`, `presentations`,
`concall_years`, `chunk_size`, `chunk_overlap`, or the resolved Qdrant collection mode. The error
names the differing key and tells the operator to start a fresh run instead. `device`,
`num_threads`, `embed_batch_size`, `workers`, `delay_seconds`, `verbose` and the notify settings
are **allowed** to differ — none of them changes the produced text or vectors, and resuming a
CPU-started run on a GPU box is a thing operators will legitimately want.

`--resume` on a `run_id` with no checkpoint directory is not an error: it warns, falls back to a
normal L1-driven run under a fresh `run_id`, and says so.

`--resume` bypasses `BatchAlreadyRunning` **only** when the recorded pid is dead per `pid_alive`
(`batch.py:469`); against a live pid it still refuses.

### 4.15 API surface

`POST /api/embeddings/nifty50/run` accepts two additive optional body keys, `resume` (bool or
run-id string) and `allow_ephemeral` (bool), passed straight through to the detached child's options
file by `write_options_file` (`batch.py:1294`). Auth, the `202 {run_id, status_url}` response, the
`409`-while-live behaviour, and the `NIFTY50_ADMIN_TOKEN` gate are unchanged.

`run_batch(options)` keeps its exact signature and return type (`RunManifest`) and becomes a thin
wrapper that builds the graph and invokes it, so the CLI, the detached child and the HTTP trigger
remain three entry points to one implementation. `process_ticker` is retained as the node bodies'
shared helper set rather than deleted wholesale — see §8 Phase 2 for the split.

---

## 5. Resulting crash-point behaviour

Compare against §2.4. "Resume" = `--resume RUN_ID`; "fresh" = a new `run_id` after Phases 1–2 ship.

| Crash during | Fresh run | `--resume` |
| :-- | :-- | :-- |
| Catalogue, ticker 40/50 | 39 catalogue calls skipped by TTL (§4.10) | 39 tickers skipped entirely (L0); no state reads |
| Download of doc *n* | completed PDFs reused; doc *n* restarts at byte 0 | same, plus earlier tickers skipped |
| Extraction of doc 2/3 | **doc 1 reused from `extracted/*.json`** (§4.9.1); doc 2 re-extracted | doc 1's subgraph already complete; doc 2 re-extracted |
| Chunking of doc *n* | extraction reused; chunk cache reused where valid | `extract` node complete, only `chunk` re-runs |
| Embedding of doc *n* | docs 1..n-1 skipped by L1/L2; doc *n*'s extraction+chunks reused, embed restarts at chunk 0 | same |
| Between `index` and `mirror` | `vectors.npz` self-heals (`batch.py:1052`) | `mirror` node re-runs alone |

Unchanged and still true: no HTTP range resume (an interrupted 40 MB report restarts at byte 0); a
single document's embedding is not resumable mid-encode; deterministic point IDs mean a re-upsert
overwrites rather than duplicates.

---

## 6. Non-goals

1. **No HTTP range/`Resume-Download` support.** A partial transfer still restarts. Adding it means
   trusting a `.part` file across process boundaries, and the validation gate in `_attempt`
   (magic bytes + readable page tree) is what makes the current design safe.
2. **No mid-document embedding resume.** `index_chunk_set` stays one embed + one upsert per
   document. Deterministic point IDs already make a repeat harmless.
3. **`index` does not move into the document subgraph.** It changes the `chunk_ready` /
   `SKIPPED -> INDEXED` reconciliation at `batch.py:1030-1035`; separate change, separate spec.
4. **No async/`ainvoke` batch.** The graph is synchronous. The saver's async methods exist so a
   future ASGI caller does not block the loop, nothing more.
5. **No LangGraph in the RAG pipeline's checkpointing.** `ingestion/rag/pipeline.py` keeps
   `compile()` with no checkpointer; a report run is minutes, not hours.
6. **No change to `state.json`'s existing keys, the manifest schema, or any REST contract**, beyond
   the additive fields named in §4.10 and §4.13.
7. **No SQL/ORM dependency**, per CLAUDE.md. `langgraph-checkpoint-sqlite` is explicitly out.
8. **No `CachePolicy` / `BaseCache`** (§4.8).

---

## 7. Verification plan

Extends `scripts/verify_nifty50_embeddings.py` (built on `scripts/checks.py`). **All offline**, no
network, no GPU, no Qdrant server — `--live SYM` keeps its existing role.

1. **`JsonFileSaver` conformance.** `put` -> `get_tuple` round-trip preserves `checkpoint`,
   `metadata` and `parent_config`. `list` honours `limit`, `before` and `filter`, newest-first.
   `put_writes` is idempotent for a repeated `(task_id, idx)`. `delete_thread` removes the
   directory and leaves sibling threads intact. Concurrent `put` from 8 threads produces no
   truncated or partial JSON.
2. **Graph resume.** Build the real graph over stub nodes with a call counter; make the third node
   raise on its first invocation. `invoke` -> catch -> `invoke(None, config)`. Assert: nodes 1–2 ran
   exactly once total, node 3 ran twice, node 4 ran once, and the final state equals a clean run's.
3. **Bounded fan-out.** With `workers=2` and 7 tickers, assert `dispatch` never emits more than 2
   `Send`s per superstep, that all 7 run exactly once, and that a checkpoint exists at each batch
   boundary. Repeat with `resolve_workers` forced to 1 via a low-VRAM `HardwareProfile` — the
   existing GPU-capping check stays green.
4. **Checkpoint size guard.** Run a ticker whose stub chunker produces 900 chunks; assert **no**
   checkpoint file exceeds 64 KB and the thread directory stays under 2 MB. This is the executable
   form of §4.3's hard rule.
5. **Extraction reuse.** Monkeypatch `Extractor.converter` with a counter. Two `run()` calls over
   one fixture PDF -> exactly one convert, and the second document equals the first. Then assert a
   re-extract when: `source_sha256` differs; `accurate_tables` flips (the `+fast-tables` version
   suffix); `max_pages` differs; `blocks` is empty. And assert **no** re-extract when only `device`
   or `num_threads` differ — the existing extractor-version/table-mode contract check is extended,
   not replaced.
6. **Catalogue TTL.** Stub `fetch_catalog` with a counter. Run twice inside the TTL against a
   fully-`INDEXED` state -> exactly one call, and a `catalog_skipped` event in `events.jsonl`. Then
   mark one document `FAILED` and assert the second run **does** call it.
7. **Ephemeral refusal.** With no `QDRANT_API_URL`, `preflight` raises `BatchError` and no
   `documents/` directory is created. With `--allow-ephemeral`, it proceeds and logs ERROR. With
   `--dry-run`, it proceeds silently.
8. **Index-failure state (P7).** Force `upsert_points` to raise; assert `state.json` records
   `status="FAILED"`, that `pending_reason` then returns `REASON_RETRY`, and that
   `should_index_document` still returns `True`.
9. **Resume guards.** `--resume` against a manifest recorded with the opposite `fast_tables` exits
   non-zero naming `fast_tables`; against a differing `device` it proceeds.
10. **`.part` sweep.** A `.part` file aged past the TTL is removed; a fresh one survives; a real
    `.pdf` is never touched.
11. **Contract regression.** `run_manifest.json` written by the graph deserialises through the
    existing `RunManifest.from_dict`, and `--status` / `status_report` produce byte-identical output
    for an equivalent pre-graph manifest.
12. **`--no-checkpoint` parity.** The same fixture run with and without the checkpointer produces
    identical `state.json`, identical manifest counters and no `checkpoints/` directory in the
    second case.

Definition of Done additions, on top of CLAUDE.md's five:

- `python scripts/verify_nifty50_embeddings.py` — zero failures, including all of the above.
- `python -m scripts.embed_nifty50 --limit 3 --dry-run`, then a real `--limit 3`, then a **killed**
  `--limit 3` (Ctrl+C mid-extraction) followed by `--resume`, then a fourth `--limit 3` confirming
  the steady-state no-op.
- `python -m scripts.embed_nifty50 --hardware` unchanged.
- `output/_nifty50/checkpoints/` is gitignored.

---

## 8. Rollout

Deliberately sequenced so the cheap, high-value half ships first and independently.

**Phase 1 — resumption economics, no LangGraph.** §4.9 (extraction + chunk cache reads, index
failure state), §4.10 (catalogue TTL), §4.11 (ephemeral refusal), §4.12 (`.part` sweep), §4.14's
`--catalog-ttl` / `--allow-ephemeral` / `--pending-only`. Verification tests 5–8, 10.

> This phase alone removes the dominant cost of a restart — the repeated Docling passes and the
> redundant catalogue requests — while `process_ticker` keeps its current shape. It is worth
> shipping and merging on its own even if Phase 2 is deferred indefinitely.

**Phase 2 — the graph.** `ingestion/graph/{state,checkpoint,nodes,build}.py`; `run_batch` becomes
the graph invoker; `--resume` / `--no-checkpoint` / `--forget-run`; API passthrough. Verification
tests 1–4, 9, 11, 12. `--no-checkpoint` is the rollback path, so Phase 2 is revertible by flag
rather than by deploy.

**Phase 3 — docs.** CLAUDE.md: the new flags in Development Commands; `GROWNXT_CATALOG_TTL_HOURS`
in Configuration; the resume/`--resume` behaviour and the four-layer model in Common Gotchas; the
`.part` sweep and the ephemeral refusal in Non-obvious Constraints.

---

## 9. Open questions

1. **`--resume` as the default on a crashed manifest?** As specced it is opt-in. Making a plain
   `python -m scripts.embed_nifty50` auto-resume a `FAILED` manifest with a live checkpoint is more
   convenient and less predictable. Recommend opt-in for the first release.
2. **Checkpoint retention.** `prune` is inherited and unused. Keep the newest *N* runs' threads, a
   time-based sweep, or unbounded until `--forget-run`? Recommend keeping the newest 5 run threads,
   swept in `preflight`.
3. **`catalog_ttl_hours` default.** 6 h is a guess: long enough to make a same-day restart free,
   short enough that a morning filing is picked up by an evening run. Depends on the intended
   schedule, which is not recorded anywhere in the repo.
4. **Does the batch actually need per-document subgraphs**, or is a checkpoint after each document
   inside one `documents` node enough? Subgraphs give cleaner namespacing and per-node retry;
   a single node with a loop gives fewer, larger checkpoints. Recommend subgraphs, but it is
   reversible.
5. **`GROWNXT_CHECKPOINT_DIR`?** Checkpoints currently live under `OUTPUT_DIR`, which is mirrored to
   Google Drive by `storage/gdrive.py` for PDFs. Confirm the mirror's scoping excludes
   `_nifty50/checkpoints/` before this generates upload traffic.

---

## 10. Implementation checklist

**Phase 1**
- [ ] Add `source_sha256` / `source_bytes` to `ExtractedDocument.meta` at write time
      (`ingestion/documents/extract.py`).
- [ ] Add `reuse: bool = True` to `Extractor.run()` with the §4.9.1 validation gate; pass
      `reuse=False` from every `write=False` caller in `scripts/verify_*.py`.
- [ ] Add the §4.9.2 chunk-cache reuse check to the chunking step.
- [ ] Record `DocumentIndexState(status="FAILED", ...)` in `index_chunk_set`'s except branch
      (`ingestion/indexer.py`).
- [ ] Add `catalog_checked_at` to `TickerState`; add `catalog_ttl_hours` to `BatchOptions` and the
      §4.10 skip logic + `catalog_skipped` event.
- [ ] Add `allow_ephemeral` and turn `batch.py:1204` into a `BatchError`.
- [ ] Add the `.part` sweep to `DocumentStore.ensure()`.
- [ ] Add `--catalog-ttl`, `--allow-ephemeral`, `--status --pending-only` to
      `scripts/embed_nifty50.py`.
- [ ] Verification tests 5, 6, 7, 8, 10.

**Phase 2**
- [ ] `ingestion/graph/state.py` — the three `TypedDict`s and the named reducers (§4.3).
- [ ] `ingestion/graph/checkpoint.py` — `JsonFileSaver` with `get_tuple`, `list`, `put`,
      `put_writes`, `delete_thread`, async delegates, atomic writes, one write lock (§4.4).
- [ ] `ingestion/graph/nodes.py` — node functions over the existing stage helpers; **no payloads in
      state**.
- [ ] `ingestion/graph/build.py` — `build_batch_graph()`, batched `dispatch` fan-out (§4.6), the
      §4.7 retry policies, the §4.8 no-`CachePolicy` comment, guarded `langgraph` import matching
      `ingestion/rag/pipeline.py`'s `LANGGRAPH_AVAILABLE` pattern.
- [ ] Rewire `run_batch` to build + `invoke(..., durability="sync")`, signature and return type
      unchanged.
- [ ] Add `--resume [RUN_ID]` with the §4.14 guards, `--no-checkpoint`, `--forget-run`.
- [ ] Thread `resume` / `allow_ephemeral` through `write_options_file` and
      `POST /api/embeddings/nifty50/run`.
- [ ] Gitignore `output/_nifty50/checkpoints/`.
- [ ] Verification tests 1, 2, 3, 4, 9, 11, 12.

**Phase 3**
- [ ] CLAUDE.md: Development Commands, Configuration table, Common Gotchas, Non-obvious Constraints
      (§8 Phase 3).
- [ ] `.env.example`: `GROWNXT_CATALOG_TTL_HOURS`.
- [ ] Smoke sequence from §7's Definition of Done, including the kill-and-`--resume` step, before
      running against all 50.
