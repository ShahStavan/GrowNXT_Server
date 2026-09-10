# Plan: Resumable Nifty 50 Embedding via a Checkpointed LangGraph Orchestrator

- **Status**: Ready for implementation, pending the 3 decisions in §C
- **Spec**: [`.claude/specs/nifty50-resumable-langgraph-orchestration.md`](../specs/nifty50-resumable-langgraph-orchestration.md)
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-03
- **Target branch**: `feat/nifty50-resumable-graph` (cut from `feat/tier2-composites-and-verification`)
- **Headline**: the spec is accurate, but **its "retained as the node bodies' shared helper set"
  line hides a full day of work** — there are no helpers, `process_ticker` is a 430-line monolith
  (§A/F1). That refactor becomes Phase 1 and is the whole answer to "modular and concise".

---

## A. Spec verification — what I checked against the code

Every load-bearing claim in the spec was re-checked against `HEAD`. These verified true:

| Spec claim | Verified |
| :--- | :--- |
| `langgraph==1.2.11` + `langgraph-checkpoint==4.2.0` already locked | ✅ `requirements.lock:80-81`; no new dependency |
| Only `checkpoint/{base,memory,serde}` installed — no `SqliteSaver` | ✅ `venv/Lib/site-packages/langgraph/checkpoint/` has exactly those three |
| `get_tuple`/`list`/`put`/`put_writes` are the four `NotImplementedError` methods | ✅ `checkpoint/base/__init__.py:239,253,277,300`; `delete_thread` at 320 also raises |
| `get_delta_channel_history` has a working default | ✅ `checkpoint/base/__init__.py:582` — documented beta, "default will continue to work" |
| `Durability = Literal["sync","async","exit"]`, default `"async"` | ✅ `langgraph/types.py:89`; default at `pregel/main.py:2603` |
| `checkpoint_during` is deprecated | ✅ `pregel/main.py:2727` emits the warning |
| `Send` has no concurrency knob | ✅ `langgraph/types.py:704`; `add_node` takes `retry_policy`/`cache_policy`/`defer` only |
| Only `cache/{base,memory,redis}` installed | ✅ rules out a durable `CachePolicy`, as §4.8 argued |
| `Extractor.run()` never reads its cache | ✅ `extract.py:285-390`; `read_extraction` (`extract.py:886`) has no batch caller |
| `write_extraction` is already atomic (temp + `replace`) | ✅ `extract.py:870-884` — cache reads are safe against a torn write |
| `pid_alive` uses the Windows handle API | ✅ `batch.py:469-512` |
| `run_manifest.json` is never read back to skip a ticker | ✅ only `load_manifest` callers are `active_run`, `status_report`, `list_run_ids` |
| `ingestion/rag/pipeline.py` compiles with no checkpointer | ✅ `pipeline.py:411` — `workflow.compile()`, bare |

Below is what the spec is **wrong about, over-specifies, or does not cover**.

### F1 — 🔴 BLOCKER for "modular": there are no helpers to build nodes on

Spec §4.15 says *"`process_ticker` is retained as the node bodies' shared helper set."* There is no
helper set. `process_ticker` is **one function, `batch.py:672-1100`, ~430 lines**, containing:

- two inline closures (`fail_ticker` at 710, `record_failure` at 845) that mutate `run` and
  `outcomes` from the enclosing scope;
- the catalogue fetch with three `except` arms and its own event emission;
- the Layer 1 diff plus the `dry_run` early return;
- a download call followed by a **nested per-document loop** doing extract *and* chunk inline
  (900-978), each with its own `try`, its own `runlog.time_stage`, its own event;
- the Layer 2 disk scan (980-1049) including the `SKIPPED -> INDEXED` reconciliation;
- the Qdrant verify, the `vectors.npz` mirror, and the terminal status arithmetic.

Graph nodes cannot be thin wrappers over that. Writing nodes against it means either duplicating
the stage logic (two implementations to keep in sync — the exact failure the first spec's §3.8
avoided) or passing `TickerRun` and `runlog` into every node and mutating shared state from
concurrent `Send` tasks.

**Decision: Phase 1 is a behaviour-neutral extraction of `ingestion/stages.py`, done before any
LangGraph code exists.** See §D. It is independently valuable — `process_ticker` drops to ~110
lines and becomes readable — and it is the only way the node bodies stay under the 25-line budget
in §D.2.

### F2 — 🟢 Good news: extraction reuse needs **no** change to outcome reconciliation

`ChunkSet.fingerprint` already hashes `params` alongside the chunk text
(`chunker.py:592-596`), and `params` carries `version` (`CHUNKER_VERSION`) **and** `extractor`
(`chunker.py:582-591`). So a reused extraction produces byte-identical chunks → an identical
fingerprint → `should_index_document` returns `False` → Layer 2 returns `SKIPPED` → and
`batch.py:1030-1035` already maps `SKIPPED` **+** `doc_id in chunk_ready` to `INDEXED`.

Spec §4.9.1 is therefore safe exactly as written, with no downstream edit. This materially
de-risks Phase 2 and should be stated in the commit message so nobody "fixes" that mapping later.

### F3 — 🟠 Spec §4.9.2 over-specifies the chunk-cache check

The spec lists five fields to compare. Because `fingerprint` hashes `params` wholesale (F2), the
check collapses to **one** comparison:

```
cached = read_chunk_cache(store.chunk_file(doc_id))
reusable = cached is not None and cached.params == expected_params(document, options)
```

where `expected_params` is a small pure function mirroring `chunker.py:582-591`. Five hand-copied
field comparisons would drift the moment someone adds a param; one dict comparison cannot.
**Amends spec §4.9.2.**

### F4 — 🟠 Do not re-hash the PDF inside the extractor

Spec §4.9.1 says compare `meta["source_sha256"]` against "sha256 of the PDF on disk" and notes
`_reuse` already calls `_hash_file`. Two problems: `_hash_file` is module-private
(`download.py:597`), and re-reading a 40 MB PDF per document per run to validate a cache is a
self-defeating optimisation.

`DownloadResult.sha256` is **already populated on both paths** — the fresh download hashes on the
way past in `_stream` (`download.py:512`), and `_reuse` hashes the on-disk file
(`download.py:399`). The value is in hand at the call site.

**Decision:** `Extractor.run()` gains `source_sha256: str = ""`; the caller passes
`result.sha256`. When it is empty (the verify harness, ad-hoc callers) the extractor skips the
hash comparison and falls back to `source_bytes` + mtime, which is enough for a local fixture.
`download.py` internals stay private. **Amends spec §4.9.1.**

### F5 — 🟠 `catalog_checked_at` couples the indexer to the catalogue

Spec §4.10 puts it on `TickerState` (`indexer.py:260-286`), which is otherwise strictly about
*vector index* state — `fingerprint`, `embedding_model`, `qdrant_collection`. A catalogue
timestamp is ingestion metadata, not index metadata.

Alternatives: (a) add the field to `TickerState` anyway; (b) a separate
`output/<T>/catalog.json`. **Recommend (a)** — `state.json` is already read once per ticker per
run and (b) doubles steady-state file IO across 50 tickers for a purity win. But it is a real
smell and needs an explicit decision (§C.2) plus a comment on the field saying why it lives there.

### F6 — 🟡 `--pending-only` cannot be computed from the manifest alone

`status_report` (`batch.py:1380`) reads only the manifest. A crashed run has **no entry at all**
for tickers it never reached, so `select(.value.status != "OK")` misses exactly the ones the
operator most needs. `--pending-only` must diff `options.resolve_constituents()` against
`manifest.tickers` and emit `never_reached ∪ not_ok`. Adds a `constituents` dependency to the
`--status` path that `status_report` deliberately does not have — so compute it in the CLI, not in
`status_report`, keeping the latter a pure manifest read (spec §4.13's contract).

### F7 — 🟡 `preflight` must not raise inside a node

Spec §4.11 puts the ephemeral-Qdrant refusal in a `preflight` node. But `run_batch` raises
`BatchError` **before** `save_manifest` today (`batch.py:1129-1133`), and the CLI depends on that
exact contract (`embed_nifty50.py:465-467`, `except batch.BatchError` → `EXIT_REFUSED`). Raising
inside a node means the manifest already exists as `RUNNING`, LangGraph wraps the exception, and
the CLI's handler stops matching.

**Decision:** the persistence gate stays in `run_batch`, before graph construction. `preflight`
keeps only hardware resolution and constituent loading. **Amends spec §4.11.**

### F8 — 🟡 `on_ticker` has no graph equivalent

The CLI prints its per-ticker table from `run_batch`'s `on_ticker` callback
(`embed_nifty50.py:459`, `batch.py:1220-1222`). In the graph only `collect` knows a ticker
finished. Keep the parameter on `run_batch`, pass it into the graph via a closure captured by
`collect`, and keep the existing `contextlib.suppress(Exception)` guard — a printing callback must
never fail a run, and now it must never fail a *superstep*.

### F9 — 🟡 Resuming reuses the run's log directory

`setup_run_logging(JOB_NAME, run_id=...)` creates `logs/nifty50/runs/<run_id>/`. A resume reuses
the `run_id`, so `run.log` and `events.jsonl` append and `summary.json` is overwritten. Append is
the right semantics — one `run_id` is one logical run — but the second `run_started` event must
carry `resumed_from=<checkpoint_id>` so the event stream is not ambiguous when read with `jq`.

### F10 — 🟡 `--resume` cannot reuse `options_from_args` as-is

`options_from_args` (`embed_nifty50.py:369-400`) builds `BatchOptions` from argv only. A resume
needs the **stored** options (from `manifest.options`) overlaid with the few keys §4.14 permits to
differ. That is a new pure function, not a tweak: `merge_resume_options(stored, requested) ->
BatchOptions | RefusalReason`. Keeping it pure is what makes the §E test-9 guard checks trivial.

---

## B. What the spec got right and needs no change

Recorded so these are not re-litigated during implementation:

- The four-layer model (§4.1) and the rule that `state.json` beats the checkpointer.
- `thread_id = run_id`, and the observation that a fresh `run_id` gets no L0 benefit — this is why
  the catalogue TTL (§4.10) exists rather than being redundant with `--resume`.
- Batched dispatch over a semaphore (§4.6). Confirmed necessary: `Send` genuinely has no worker
  knob and `add_node` exposes none.
- `durability="sync"` (§4.5). The default `"async"` loses the last superstep, which is the whole
  scenario.
- No `CachePolicy` (§4.8). Confirmed by the installed cache backends.
- Narrow `RetryPolicy` (§4.7), and specifically **no** retry on `download` — `Downloader.fetch`
  already owns backoff and the `_Transient`/`_Permanent` split (`download.py:266-303`).
- The "no payloads in graph state" rule (§4.3), and testing it with a size assertion.

---

## C. Decisions needed before Phase 1

### C.1 — Is Phase 1 (the stage refactor) in scope? **Blocks everything**

F1 makes it a prerequisite for clean nodes, and it touches the single most load-bearing function
in the batch pipeline. It is behaviour-neutral and covered by the existing 77-check verify suite
plus a golden-manifest diff (§E.0), but it is still a day of work on code that currently works.

- **(a) Yes, refactor first** — recommended. Nodes stay ≤25 lines, one implementation of each
  stage, `process_ticker` becomes readable, and `--no-checkpoint` parity is free.
- (b) Skip it, write fat nodes — delivers the graph sooner and permanently duplicates stage logic.
- (c) Ship Phase 2+3 only (the artifact caches and TTL), defer the graph indefinitely. **This is
  the best value-per-risk option if time is short** — it removes the dominant restart cost without
  touching orchestration at all.

### C.2 — Where does `catalog_checked_at` live? (F5)

`TickerState` in `indexer.py` (recommended, cheapest) vs a separate `output/<T>/catalog.json`.

### C.3 — `catalog_ttl_hours` default

Spec §9 guessed 6 h. Needs the intended run schedule, which is recorded nowhere in the repo. If
the batch runs nightly, 6 h is inert and 20 h is the useful value; if it runs on demand after a
crash, 6 h is right. **Recommend 6 h until a schedule exists**, since the only current caller is
manual.

---

## D. Target module design

### D.1 — File map with responsibility and line budget

```
ingestion/stages.py                      NEW, Phase 1        ~280 lines
    The seven stages as free functions. Knows: catalog, download, extract,
    chunk, indexer, DocumentStore, RunLogger. Knows nothing about TickerRun,
    RunManifest, BatchOptions-as-a-whole, or langgraph.

    @dataclass(frozen=True) StageContext      # symbol, options-derived scalars,
                                              # store, indexer, extractor, runlog
    @dataclass CatalogueResult / DiffResult / DownloadStageResult
    @dataclass ExtractResult / ChunkResult / IndexStageResult / MirrorResult

    fetch_catalogue(ctx)                  -> CatalogueResult
    compute_diff(ctx, entries)            -> DiffResult          # wraps compute_pending
    download_pending(ctx, pending)        -> DownloadStageResult
    extract_one(ctx, doc)                 -> ExtractResult
    chunk_one(ctx, doc, document)         -> ChunkResult
    index_chunk_files(ctx, forced_ids)    -> IndexStageResult
    mirror_vectors(ctx, indexed_any)      -> MirrorResult

ingestion/batch.py                       MODIFIED             process_ticker 430 -> ~110
    Keeps: BatchOptions, DocumentOutcome, TickerRun, RunManifest, manifest IO,
    single-flight (pid_alive/is_stale/active_run), select_entries,
    pending_reason, compute_pending, resolve_workers, launch_background,
    status_report, admin_token_ok.
    process_ticker becomes: build StageContext -> call stages in order ->
    fold results into TickerRun. No inline closures; `record_failure` becomes
    a small module-level function taking (run, outcomes, ...) explicitly.
    run_batch keeps its signature and return type; body delegates to the graph.

ingestion/graph/__init__.py              NEW, Phase 4         ~25
    build_batch_graph, JsonFileSaver, checkpoint_root. Nothing else public.

ingestion/graph/state.py                 NEW, Phase 4         ~95
    BatchGraphState / TickerGraphState / DocumentGraphState TypedDicts,
    plus merge_results, merge_outcomes, sum_counters. Zero imports from
    ingestion.batch or langgraph.graph. No logic beyond the reducers.

ingestion/graph/checkpoint.py            NEW, Phase 4        ~230
    JsonFileSaver(BaseCheckpointSaver) and nothing else. Generic: no ticker,
    no manifest, no BatchOptions. Could be lifted to another project verbatim.

ingestion/graph/nodes.py                 NEW, Phase 5        ~210
    One function per node. Each reads state, calls exactly one stages.py
    function, returns a dict delta. Hard budget: 25 lines per node.

ingestion/graph/build.py                 NEW, Phase 5        ~130
    Topology, RetryPolicy attachment, batched dispatch, the no-CachePolicy
    comment, the guarded langgraph import. No business logic.

ingestion/documents/extract.py           MODIFIED, Phase 2   +~45
ingestion/documents/storage.py           MODIFIED, Phase 3   +~25   (.part sweep)
ingestion/indexer.py                     MODIFIED, Phase 2/3 +~30   (P7, catalog_checked_at)
scripts/embed_nifty50.py                 MODIFIED, Phase 3/6 +~120  (6 flags)
api/embeddings.py                        MODIFIED, Phase 6   +~10   (2 body keys)
scripts/verify_nifty50_embeddings.py     MODIFIED, all       +~350
```

### D.2 — Clean-code rules, as acceptance criteria

Not aspirations — each is checkable in review, and three are checkable by the suite.

1. **Nodes are adapters.** A node reads state, calls **one** `stages.py` function, returns a
   dict delta. **≤25 lines each, no exceptions.** If a node wants a loop, the loop belongs in
   `stages.py`. Grep-checkable.
2. **`stages.py` never imports `langgraph`.** This is what makes `--no-checkpoint` a real
   fallback, keeps the existing verify suite valid, and lets stages be tested without a graph.
   Enforced by §E test 13.
3. **`checkpoint.py` never imports `ingestion.batch`, `ingestion.stages`, or `core.config`.**
   A saver that knows about tickers is a saver that cannot be tested in isolation. Enforced by
   §E test 13.
4. **No payloads in graph state** (spec §4.3). Enforced by §E test 4's size assertion.
5. **One frozen result dataclass per stage.** No `dict[str, Any]` crossing a module boundary.
   Matches the existing `DownloadResult` / `IndexResult` convention rather than inventing one.
6. **Stage failures are returned, not raised.** Each result carries `ok` and `error`, like
   `DownloadResult` (`download.py:126`). The only exceptions that escape a stage are the ones
   `RetryPolicy` should see (§4.7).
7. **Reducers are named module-level functions.** Not lambdas — they must be importable for
   concurrent `Send` writes to fold, and testable directly.
8. **No new exception types.** `BatchError` / `BatchAlreadyRunning` / `ExtractionError` /
   `CatalogError` already cover the space.
9. **No closures over mutable run state.** F1's `fail_ticker` / `record_failure` become module
   functions with explicit parameters. Concurrent `Send` tasks make enclosing-scope mutation a
   data race waiting to happen.
10. **Google style + full type annotations + docstrings**, per CLAUDE.md. Every new public
    function gets an `Args:`/`Returns:`/`Raises:` docstring; every module gets the
    "Google Python Style Guide Compliant." footer the existing modules carry.

### D.3 — What `process_ticker` looks like after Phase 1

Structure only, to fix the target shape:

```
def process_ticker(member, options, indexer, runlog, position=0, total=0) -> TickerRun:
    ctx = StageContext.build(member, options, indexer, runlog)
    run = TickerRun(...); runlog.event("ticker_started", ...)

    cat = stages.fetch_catalogue(ctx)
    if not cat.ok:
        return _fail_ticker(run, ctx, cat)

    diff = stages.compute_diff(ctx, cat.entries)
    _record_unchanged(run, diff)
    if options.dry_run:
        return _finish_dry_run(run, ctx, diff)

    dl = stages.download_pending(ctx, diff.pending)
    for doc in dl.ready:
        ex = stages.extract_one(ctx, doc)
        if not ex.ok:
            _record_failure(run, diff.outcomes, doc.doc_id, "extract", ex.error); continue
        ch = stages.chunk_one(ctx, doc, ex.document)
        if not ch.ok:
            _record_failure(run, diff.outcomes, doc.doc_id, "chunk", ch.error); continue

    idx = stages.index_chunk_files(ctx, forced_ids=diff.forced_ids)
    _fold_index_results(run, diff.outcomes, idx)
    stages.mirror_vectors(ctx, indexed_any=idx.indexed_any)
    return _finalise(run, diff, idx)
```

Every line of the current 430 lands in exactly one of: a `stages.py` function (the work), a
`_record_*`/`_fold_*` helper (the bookkeeping), or `runlog` calls inside the stages (the
observability). Nothing is deleted and nothing changes behaviour — which is what makes §E.0
a valid gate.

---

## E. Implementation phases

Phases 1–3 are the spec's Phase 1 and are **independently shippable**. Phases 4–6 are the graph.

### Phase 0 — Branch & housekeeping (10 min)

- Cut `feat/nifty50-resumable-graph`.
- `.gitignore`: `output/_nifty50/checkpoints/`.
- Confirm `storage/gdrive.py`'s mirror scoping excludes `_nifty50/` (spec §9 open question 5) —
  a one-grep check, not a change, unless it doesn't.

### E.0 — The parity gate (applies to Phase 1)

Before touching `process_ticker`, capture a baseline: run `--limit 3` against a fixture and save
`run_manifest.json` + all three `state.json` files. After the refactor, re-run and assert
**byte-identical** `state.json` and a manifest differing only in timestamps, elapsed values and
`run_id`. Add it to the verify suite as a fixture-backed check so it guards future refactors too.
This is the only thing that makes a 430-line extraction safe.

### Phase 1 — `ingestion/stages.py` extraction (~1 day) — *the modularity phase*

1. Create `ingestion/stages.py` with `StageContext` and the seven result dataclasses (§D.1).
2. Move each block of `process_ticker` into its stage function **verbatim**, including its
   `runlog.event` / `runlog.add` / `runlog.time_stage` calls. Do not improve anything yet.
3. Convert `fail_ticker` / `record_failure` to module-level `_fail_ticker` / `_record_failure`
   with explicit parameters (rule D.2.9).
4. Rewrite `process_ticker` to §D.3's shape.
5. Gate: §E.0 parity + the existing `scripts/verify_nifty50_embeddings.py` (77 checks) green.

**Commit separately from everything else.** A behaviour-neutral refactor with its own parity
proof is reviewable; bundled with feature work it is not.

### Phase 2 — L3 artifact caches + P7 (~half day)

1. `extract.py`: add `source_sha256` / `source_bytes` to `meta` at write time (extend the dict at
   `extract.py:373-384`, and the matching dict in `_extract_fast_transcript`).
2. `extract.py`: add `reuse: bool = True` and `source_sha256: str = ""` to `Extractor.run()`;
   a private `_cached(...) -> ExtractedDocument | None` implementing spec §4.9.1's gate as amended
   by F4. Place the check **before** the transcript fast path so both paths benefit.
3. `scripts/verify_chunker.py`, `scripts/verify_documents.py`, `evaluate_full_pipeline.py`: pass
   `reuse=False` wherever they pass `write=False` (spec §4.9.1).
4. `stages.chunk_one`: the F3 one-line params comparison, plus `expected_params()` as a pure
   function beside it.
5. `indexer.py`: write `DocumentIndexState(status="FAILED", ...)` in `index_chunk_set`'s except
   branch (P7). `DocumentIndexState` already has `status` and `error` fields
   (`indexer.py:225-240`) — no dataclass change needed.
6. Gate: §E tests 5, 8 + suite green + F2 asserted (a reused extraction still yields `INDEXED`,
   not `SKIPPED`, in the `TickerRun`).

### Phase 3 — TTL, refusal, sweep, and the first four flags (~half day)

1. `indexer.py`: `TickerState.catalog_checked_at: str = ""` (per C.2), with the F5 comment.
2. `BatchOptions`: `catalog_ttl_hours: float = 6.0`, `allow_ephemeral: bool = False`.
3. `stages.fetch_catalogue`: the §4.10 skip gate + `catalog_skipped` event; write the timestamp
   after every successful fetch.
4. `run_batch`: turn `batch.py:1204-1207` into a `BatchError` unless `allow_ephemeral` or
   `dry_run` — **in `run_batch`, before the graph exists** (F7).
5. `storage.py`: `.part` sweep in `ensure()` with `PART_FILE_TTL_HOURS = 6`.
6. `embed_nifty50.py`: `--catalog-ttl`, `--allow-ephemeral`, `--status --pending-only` (computing
   the never-reached set per F6), and the `GROWNXT_CATALOG_TTL_HOURS` read.
7. Gate: §E tests 6, 7, 10 + suite green.

> **Phases 1–3 are a coherent, mergeable PR.** They remove the repeated Docling passes and the
> redundant catalogue requests — the dominant restart cost — with no LangGraph code. If C.1
> resolves to (c), stop here.

### Phase 4 — Graph state + the checkpointer (~1 day)

1. `ingestion/graph/state.py`: the three `TypedDict`s and three named reducers (§D.1). Reducers
   first, with direct tests — they are the only place a concurrent-write bug can hide.
2. `ingestion/graph/checkpoint.py`: `JsonFileSaver`. Order of work:
   - the path layout + atomic-write helper (reuse `save_ticker_state`'s pattern, `indexer.py:492`);
   - `put` then `get_tuple` (round-trip first, so test 1 can drive the rest);
   - `put_writes` with `(task_id, idx)` idempotency;
   - `list` with `filter`/`before`/`limit`;
   - `delete_thread`;
   - the async delegates via `asyncio.to_thread`;
   - one `threading.Lock` around the write path.
   Do **not** override `get_delta_channel_history`, `prune`, `delete_for_runs`, `copy_thread`.
3. Gate: §E tests 1, 4, 13.

### Phase 5 — Nodes, topology, and rewiring `run_batch` (~1 day)

1. `ingestion/graph/nodes.py`: one node per stage, each ≤25 lines, each calling exactly one
   `stages.py` function (this is only possible because of Phase 1).
2. `ingestion/graph/build.py`: the §4.2 topology, batched `dispatch` (§4.6) reusing
   `resolve_workers` verbatim, the §4.7 retry policies, the guarded import mirroring
   `pipeline.py:26-32`, and the §4.8 comment.
3. `batch.py`: `run_batch` keeps its signature and `RunManifest` return; body becomes preflight →
   build → `invoke(..., durability="sync")` → manifest. `on_ticker` threaded into `collect`
   via closure with the existing suppression (F8). Keep `process_ticker` — it is now the
   `--no-checkpoint` path and the stage-composition reference.
4. `collect`: write the manifest projection (spec §4.13); add `resumable` / `resumed_from`.
5. Gate: §E tests 2, 3, 11, 12.

### Phase 6 — Resume surface + API (~half day)

1. `merge_resume_options(stored, requested)` as a pure function (F10) implementing §4.14's
   allow/refuse split.
2. `--resume [RUN_ID]`, `--no-checkpoint`, `--forget-run RUN_ID`.
3. `run_started` carries `resumed_from` (F9).
4. `api/embeddings.py`: `resume: bool | str = False` and `allow_ephemeral: bool = False` on
   `EmbeddingRunRequest`, passed through `write_options_file`. Auth, `202`, `409` unchanged.
5. Gate: §E test 9 + a manual kill-and-resume.

### Phase 7 — Verification & docs (~half day)

All 13 checks green; CLAUDE.md, `.env.example`; the §F smoke sequence.

---

## F. Verification plan

Extends `scripts/verify_nifty50_embeddings.py` via `scripts/checks.py`'s `Report`
(`report.section` / `report.check` / `report.raises` / `report.run` / `report.finish`). All
offline. Spec §7's 12 tests, plus:

- **Test 0 (new)** — the §E.0 parity gate: fixture run before/after Phase 1 produces identical
  `state.json` and a manifest differing only in timestamps.
- **Test 13 (new)** — import hygiene, enforcing D.2.2 and D.2.3: assert `langgraph` is absent from
  `ingestion/stages.py`'s imports, and that `ingestion/graph/checkpoint.py` imports nothing from
  `ingestion.batch`, `ingestion.stages`, or `core.config`. A five-line AST walk; it is the only
  automated defence of the module boundaries.
- **Node line budget** — folded into test 13: assert no function in `nodes.py` exceeds 25
  statements (`ast.walk` + `len(node.body)`).

Definition of Done, on top of CLAUDE.md's five:

1. `python scripts/verify_nifty50_embeddings.py` — zero failures.
2. `python scripts/verify_chunker.py`, `verify_documents.py`, `verify_reporting.py` — unchanged and
   green (Phase 2 touches their `Extractor` calls).
3. `python -m scripts.embed_nifty50 --limit 3 --dry-run` → real `--limit 3` → **Ctrl+C mid-extraction**
   → `--resume` → a fourth `--limit 3` confirming the steady-state no-op.
4. `python -m scripts.embed_nifty50 --hardware` unchanged.
5. `python -m scripts.embed_nifty50 --limit 3 --no-checkpoint` matches (3)'s `state.json`.

---

## G. Risks

| Risk | Mitigation |
| :--- | :--- |
| **Phase 1 changes behaviour silently.** 430 lines moved by hand. | §E.0 parity gate + 77 existing checks, committed separately with its own diff. Highest-risk item in the plan. |
| `JsonPlusSerializer` chokes on a state value. | D.2.4 keeps state to scalars/paths; test 4's size assertion catches payload creep early. |
| Checkpoint file count grows unbounded (50 tickers × ~6 nodes × supersteps). | Spec §9 Q2: prune to the newest 5 run threads in `preflight`. Decide during Phase 4. |
| `Send` fan-out surprises — task ordering, reducer folding under concurrency. | Reducers tested directly before any graph exists (Phase 4 step 1); test 3 pins batch boundaries. |
| `--resume` resumes into a corrupt checkpoint. | `get_tuple` returns `None` on unparseable JSON → falls back to a fresh L1 run and warns (spec §4.14). Never raise from a read. |
| LangGraph minor-version drift changes `BaseCheckpointSaver`. | Version is pinned in `requirements.lock` (1.2.11 / checkpoint 4.2.0). Test 1 is the canary; only the four documented `NotImplementedError` methods are overridden. |
| Scope creep into `ingestion/rag/pipeline.py`. | Spec §6.5: out of scope. A report run is minutes. |

---

## H. Checklist

**Phase 0**
- [ ] Branch `feat/nifty50-resumable-graph`; gitignore `output/_nifty50/checkpoints/`.
- [ ] Confirm `storage/gdrive.py` mirror excludes `output/_nifty50/`.

**Phase 1 — modularity (own commit)**
- [ ] Capture the §E.0 baseline fixture (manifest + 3 `state.json`).
- [ ] `ingestion/stages.py`: `StageContext` + 7 result dataclasses + 7 stage functions.
- [ ] `_fail_ticker` / `_record_failure` / `_record_unchanged` / `_fold_index_results` /
      `_finalise` as module functions (no closures).
- [ ] `process_ticker` rewritten to §D.3, ≤120 lines.
- [ ] Test 0 + full suite green.

**Phase 2 — artifact caches**
- [ ] `source_sha256` / `source_bytes` in `ExtractedDocument.meta`, both extraction paths.
- [ ] `Extractor.run(reuse=True, source_sha256="")` + `_cached()` gate (F4).
- [ ] `reuse=False` at every `write=False` call site in `scripts/`.
- [ ] `expected_params()` + one-line chunk-cache comparison (F3).
- [ ] `DocumentIndexState(status="FAILED", ...)` on index failure (P7).
- [ ] Tests 5, 8; F2 assertion.

**Phase 3 — TTL, refusal, sweep**
- [ ] `TickerState.catalog_checked_at` (per C.2) + `catalog_skipped` event + skip gate.
- [ ] `catalog_ttl_hours`, `allow_ephemeral` on `BatchOptions`.
- [ ] Ephemeral refusal in `run_batch`, not a node (F7).
- [ ] `.part` sweep in `DocumentStore.ensure()`.
- [ ] `--catalog-ttl`, `--allow-ephemeral`, `--status --pending-only` (F6).
- [ ] Tests 6, 7, 10.

**Phase 4 — state + saver**
- [ ] `graph/state.py` TypedDicts + 3 named reducers, reducers tested first.
- [ ] `graph/checkpoint.py` `JsonFileSaver`: `put`, `get_tuple`, `put_writes`, `list`,
      `delete_thread`, async delegates, atomic writes, one lock.
- [ ] No override of `get_delta_channel_history` / `prune` / `delete_for_runs` / `copy_thread`.
- [ ] Tests 1, 4, 13.

**Phase 5 — nodes + topology**
- [ ] `graph/nodes.py`, every node ≤25 statements, one stage call each.
- [ ] `graph/build.py`: topology, batched dispatch via `resolve_workers`, retry policies,
      guarded import, no-`CachePolicy` comment.
- [ ] `run_batch` rewired, signature and return type unchanged; `on_ticker` via `collect` (F8).
- [ ] `collect` writes the manifest projection + `resumable` / `resumed_from`.
- [ ] Tests 2, 3, 11, 12.

**Phase 6 — resume surface**
- [ ] `merge_resume_options` (F10) + `--resume [RUN_ID]` / `--no-checkpoint` / `--forget-run`.
- [ ] `run_started` carries `resumed_from` (F9).
- [ ] `resume` / `allow_ephemeral` on `EmbeddingRunRequest` → `write_options_file`.
- [ ] Test 9 + manual kill-and-resume.

**Phase 7 — docs**
- [ ] CLAUDE.md: Development Commands, Configuration (`GROWNXT_CATALOG_TTL_HOURS`), Common
      Gotchas (the four-layer model, `--resume` vs `--force`), Non-obvious Constraints (`.part`
      sweep, ephemeral refusal, the extraction-cache reuse key).
- [ ] `.env.example`: `GROWNXT_CATALOG_TTL_HOURS`.
- [ ] §F smoke sequence, then a full 50-ticker run.
