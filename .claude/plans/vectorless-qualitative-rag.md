# Plan: Replace the Vector Pipeline with Vectorless Qualitative RAG

- **Status**: Ready for implementation, pending the 4 decisions in §C
- **Spec**: [`.claude/specs/vectorless-qualitative-rag.md`](../specs/vectorless-qualitative-rag.md)
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-10
- **Target branch**: `feat/vectorless-qualitative-rag`, cut from `feat/nifty50-resumable-graph`
- **Scope change against the spec**: the user has directed **full replacement**. Spec §5
  (`GROWNXT_RETRIEVAL_MODE`, vector stack retained behind a flag) and the hedged phasing in
  spec §9 are **superseded by this plan**. There is no `hybrid` mode and no `vector` mode.
- **Headline**: two load-bearing claims in the spec are **wrong** and the plan corrects them
  (§A/F1, §A/F2). Torch does not leave the project — Docling owns it. And the GPU worker cap
  stays, because it was never about the embedder. What actually leaves is
  `sentence-transformers`, `qdrant-client`, and 3,192 lines of retrieval code, against
  ~2,430 lines of new code.

---

## A. Verification against the code

Every load-bearing claim in the spec was re-checked against `HEAD` of
`feat/nifty50-resumable-graph`. These verified true:

| Spec claim | Verified |
| :--- | :--- |
| `financial_page_range()` returns `(start, total)` — keeps the tail | ✅ `sections.py:141` |
| ADANIENT FY2026: 396 pages, 2,106,468 chars, 11,260 blocks, 626 tables | ✅ `output/ADANIENT/extracted/annual_report_FY2026.json` meta |
| MD&A at p158/166, Board's Report p154, BRSR p193, KAM p222/224 | ✅ heading scan over that extraction |
| Financial statements begin p222; anchor fires there | ✅ first anchor-matching heading page = 222 |
| Chunk JSON for that filing is 7.7 MB | ✅ `output/ADANIENT/chunks/annual_report_FY2026.json` = 7,746,511 B |
| Extraction `sections` list contains junk (`'2.'`, `'~7 million 3'`) | ✅ `meta.sections` |
| Retrieval issues 15 static queries from `probes.py` | ✅ 5 pillars × 3 queries, `probes.py:60-140` |
| `Extractor._cache_key` already takes `page_range` | ✅ `extract.py:312,347` |
| `extract_version()` already folds text-changing settings | ✅ `extract.py:128-150` |
| `document_reason` is the single definition of "current" | ✅ `stages.py:337`, called by `pending_reason:372` only |
| `page_range` participates in the cache key, device does not | ✅ `extract.py:326-349` |
| Graph state carries identifiers and counts, never payloads | ✅ `graph/state.py` docstring + `BatchGraphState` |
| Batch path already runs `ocr=False, figures=False` | ✅ `stages.py:696` `build_extractor` |

### F1 — 🔴 The spec is wrong: torch does not leave. Docling owns it.

Spec §7 claims the embedding model and GPU are "removed", and §1's success criterion says
"zero GPU". That is true of the **query** path and false of the **ingest** path.

```
docling-ibm-models==4.0.1  requires  torch<3.0.0,>=2.2.2
                                     torchvision<1,>=0
                                     transformers>=4.42.0,<6.0.0
```

Docling's layout model and TableFormer are torch models. `requirements.lock:174` pins
`torch==2.13.0` and it stays pinned. **Correction to carry into the spec**: what removal buys is
`sentence-transformers==6.0.1`, `qdrant-client==1.19.0`, the 400 MB pre-cached Arctic layer in the
Dockerfile, and — a real bonus — the relaxation of a genuine version conflict, since
`sentence-transformers` demands `transformers>=5.0.0` while `docling-ibm-models` demands `<5.9.0`.

Rewrite the success criterion as: **zero GPU and zero model weights in the query path; Docling's
torch dependency unchanged in the ingest path.**

### F2 — 🔴 The GPU worker cap stays. It was never about the embedder.

`CLAUDE.md` and `resolve_workers()` cap `--workers` to 1 under 8 GB VRAM because "workers are
threads sharing a process, so each puts its own **layout and table models** on the same card".
Those are Docling's models. The cap is unaffected by removing the embedder and **must not be
deleted** as part of this work.

By the same argument `core/hardware.py` is **trimmed, not removed**: `device`, `num_threads`,
`idle_gpu` and `resolve_workers` all still serve Docling. Only `embed_batch_size` and
`embed_fp16` become dead.

### F3 — 🟢 The consumer seam is a single function

`reporting/engine.py:134` is the **only** external caller of the RAG stack:

```python
from ingestion.rag.pipeline import extract_ticker_findings

dossier = extract_ticker_findings(ticker=symbol, company_name=symbol, force=refresh)
```

`extract_ticker_findings(ticker, company_name, force) -> ResearchDossier` is therefore the one
signature that must survive byte-for-byte. Everything behind it is free to be replaced. This is
much better than feared and it is what makes a full replacement tractable in one branch.

### F4 — 🟠 `StageContext.indexer` is the real blast radius

`StageContext` is `@dataclass(frozen=True)` with `indexer: QdrantVectorIndexer` (`stages.py:127`),
and `GraphResources.indexer` mirrors it (`nodes.py`). Swapping the type touches:

- `ingestion/stages.py` — 6 sites (`:48-51, :127, :135, :375, :409, :882`)
- `ingestion/batch.py` — 8 sites (`:76, :634, :892, :1096, :1172, :1217, :1259, :1265`)
- `ingestion/graph/nodes.py` — `GraphResources.indexer`, `run_ticker`
- `scripts/verify_nifty50_embeddings.py` — ~20 sites

Mitigation: name the replacement `LedgerStore` and give it the three methods the context actually
uses — `load_ticker_state`, `save_ticker_state`, `collection_for` → `namespace_for`. Then the
rename is mechanical and reviewable in one commit, separate from behaviour change (Phase 1).

### F5 — 🟠 `ingestion/__init__.py` re-exports the entire surface being deleted

Lines `14-24` (chunker), `58-68` (indexer), `70-81` (rag), and `__all__` at `101-138` name
`ChunkSet`, `chunk_document`, `CHUNKER_VERSION`, `QdrantVectorIndexer`, `IndexerConfig`,
`index_ticker_documents`, `ParallelVectorRetriever` and more. Every deletion breaks this file
first. It is also the cheapest possible smoke test: `python -c "import ingestion"` fails loudly
the moment a removal is incomplete. Make it the parity gate for every phase.

### F6 — 🟢 No `state.json` migration script is needed

`document_reason` returns `REASON_VERSION` whenever a recorded version field differs from the
run's. An existing `output/<TICKER>/state.json` has no `qualitative_version`, so every document
in it re-processes on first run — which is exactly right, and it self-heals with no migration
code. The dead `qdrant_collection` field is simply ignored by `from_dict`.

The one thing to get right: reprocessing must **reuse the Docling extraction on disk** when
`extract_version` still matches, so the migration costs SLM calls and not a re-extraction.
That is §E/Phase 4's acceptance test, and it is the single most expensive thing to get wrong.

### F7 — 🟠 `verify_nifty50_embeddings.py` must be split, not deleted

2,350+ lines, and its coverage divides cleanly:

- **Still valid** — Layer 1 diff, `document_reason` inversion guard, catalogue TTL,
  `pid_alive`, `_check_launch_handoff`, hardware resolution, `resolve_workers` capping,
  extractor-version/table-mode contract, `check_page_filter`'s auditor-certificate guard.
- **Dead** — everything asserting embeddings, Qdrant collections, `embed_texts` call counts
  (`:2320-2332`), vector mirrors, `IndexerConfig.prefer_grpc` (`:902-924`).

Keep the file and its name (the CLAUDE.md-documented command stays working), delete the dead
checks, and add the new ones in `scripts/verify_qualitative_rag.py`.

### F11 — 🟠 The tree already fails 4 gate checks. Baseline recorded 2026-09-10.

Phase 0 ran `scripts/parity_gate.py` before any replacement work. **19 of 23 pass.** None of
the four failures is caused by this work, and each has to be classified now, because after
Phase 2 it becomes impossible to tell a pre-existing failure from one this work introduced.

| Failure | Cause | Disposition |
| :--- | :--- | :--- |
| `import app` — `TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'` | `requirements.lock:48` pins **gradio 6.26.0** while `requirements.txt` says `>=5.16.0` and the code targets Gradio 5. `gr.Chatbot(type=...)` (`app.py:358`) was removed in Gradio 6. | **Out of scope, but blocks Definition of Done #4** — the Space entry point cannot boot in this venv. Raise separately. |
| `verify_chunker.py` — `ModuleNotFoundError: No module named 'ingestion'` | The script has **no `sys.path` bootstrap**, unlike `verify_documents.py:53`. The CLAUDE.md-documented `python scripts/verify_chunker.py` has been broken; only `python -m scripts.verify_chunker` works. | Moot — Phase 5 deletes this file. Do not spend time on it. |
| `verify_documents.py` — 2 of 51: "a heading was found" (0 headings), "blocks carry a heading trail" | Docling heading detection on the synthetic fixture, not on real filings — the ADANIENT extraction has 2,949 headings. | **Watch closely.** Phase 2's `sectionmap.py` depends entirely on Docling emitting headings. Confirm the fixture is at fault, not the detector, before Phase 2 builds on it. |
| `verify_reporting.py` — 1 of 7: "Composites carry components: no cached tickers were available" | Environmental; needs cached collector JSON in `output/`. | Pre-existing, environmental. |

So the gate's Phase 0 baseline is **19/23**, and every later phase must hold at 19/23 or
better — never "green", which this tree has never been.

### F8 — 🟡 The API contract must accept dead fields, not reject them

`api/embeddings.py:52-56` accepts `device`, `num_threads`, `embed_batch_size`, `fast_tables`,
`page_filter`. Under replacement, `embed_batch_size` is meaningless and `page_filter` inverts.
Definition of Done #5 requires JSON schemas to be preserved, so: **keep both fields in the
model, ignore `embed_batch_size` with a one-line deprecation warning, and map `page_filter` to
the new `qualitative_filter`** rather than 400-ing. Same for the CLI's `--embed-batch-size`.

### F9 — 🟡 `llama-index-core` stays, and earns its place

The new ledger extractor must window a 95k-char MD&A into SLM-sized units. `SentenceSplitter` is
already locked, already used, and correct for that. Keeping it means `chunker.py` dies without a
new splitter having to be written. Only the *document-shaped chunking policy* is being replaced,
not sentence splitting itself.

### F10 — 🟠 Three scripts import the doomed modules; my first audit truncated and missed two

A ripgrep pass with a result limit hid these. A full scan found them:

| Script | State today | Action |
| :--- | :--- | :--- |
| `scripts/evaluate_full_pipeline.py` | **Live** — imports cleanly, uses `chunk_document` + `write_chunk_cache` (`:22-27, :133, :141`), and is documented in CLAUDE.md's testing section | **Rewire in Phase 5**, not delete |
| `scripts/ingest_documents.py` | Already dead — `ImportError: cannot import name 'DEFAULT_ANNUAL_SKIP_SECTIONS' from 'ingestion.chunker'` | Delete in Phase 5 |
| `scripts/verify_ingestion.py` | Already dead — `ModuleNotFoundError: No module named 'ingestion.embedder'` (`ingestion.layout`, `ingestion.parsers` also gone) | Delete in Phase 5 |

The two dead ones are exactly what CLAUDE.md's "Testing & Verification" NOTE warns about; they
have been broken since the Docling refactor. Deleting them alongside `chunker.py` removes the
note as well.

`evaluate_full_pipeline.py` is the gap that matters: it is a working, documented command that
Phase 5's deletion of `chunker.py` breaks. It runs
extract → chunk → write cache, so the rewire is
extract → `map_sections` → `build_ledger`, and it becomes a useful end-to-end smoke test of the
new pipeline rather than a casualty of it.

**Consequence for the parity gate**: `python -c "import ingestion"` (F5) does **not** cover
`scripts/`. The gate needs an explicit script-import sweep, or a break like this stays invisible
until someone runs the command by hand.

---

## B. What replacement means, concretely

| | Today | After |
| :--- | :--- | :--- |
| Retrieval structure | Qdrant per-ticker collection, 768-dim Arctic | `facts.jsonl` + `lexical.npz`, filter + BM25 |
| Ingest unit | 800-char `SentenceSplitter` window | classified `SectionSpan` / `SpeakerTurn` / `Slide` |
| Retrieval unit | `EvidenceChunk` (text + score) | `QualitativeFact` (typed claim + verbatim quote + page) |
| Page selection | financial tail, p222-396 | qualitative spans, p2-221 + KAM + 4 note classes |
| Query | 15 embedded probe queries | 6 declarative `LedgerQuery` filters |
| Ingest models | Docling (torch) + Arctic (torch/GPU) | Docling (torch) + hosted SLM (HTTP) |
| Query models | Arctic encode + reranker + SLM | SLM only |
| Per-ticker artefacts | `chunks/`, `vectors.npz`, `payloads.json`, `state.json` | `qualitative/{sectionmap,facts.jsonl,lexical.npz,vocab}.json`, `state.json` |

**Deleted outright** — 3,192 lines:

| File | Lines | Replaced by |
| :--- | ---: | :--- |
| `ingestion/indexer.py` | 1,617 | `qualitative/store.py` (state) + `qualitative/ledger.py` (records) |
| `ingestion/chunker.py` | 677 | `qualitative/sectionmap.py` + `segments.py` |
| `ingestion/rag/retriever.py` | 511 | `qualitative/query.py` |
| `ingestion/rag/reranker.py` | 200 | ranking folded into `qualitative/lexical.py` |
| `ingestion/rag/probes.py` | 187 | `qualitative/routing.py` |
| `scripts/verify_chunker.py` | — | `scripts/verify_qualitative_rag.py` |
| `scripts/ingest_documents.py` | — | already dead (F10); deleting it retires CLAUDE.md's ImportError note |
| `scripts/verify_ingestion.py` | — | already dead (F10) |

Also deleted: `output/<T>/chunks/`, `vectors.npz`, `payloads.json`,
`ingestion/rag/probe_vectors.json`, the Dockerfile's Arctic pre-cache layer, and
`QDRANT_*` from `.env.example` and the CLAUDE.md variable table.

**Kept and rewired:**

- `ingestion/rag/pipeline.py` — the LangGraph shape and `ResearchState` survive; the
  `parallel_retrieve` node's body is replaced and `rerank_evidence` is dropped from the topology.
- `ingestion/rag/synthesizer.py` — `_build_evidence_context` takes facts instead of chunks;
  `ThematicFinding` / `ResearchDossier` schemas unchanged, so `findings.json` stays compatible.
- `ingestion/stages.py` — `chunk_one` → `map_sections`, `index_chunk_files` → `build_ledger`,
  `verify_and_mirror` → `write_lexical_index`.
- `ingestion/batch.py`, `ingestion/graph/nodes.py` — `indexer` → `store` (F4).
- `scripts/evaluate_full_pipeline.py` — extract → chunk → cache becomes
  extract → `map_sections` → `build_ledger` (F10). Live and documented; must not be collateral.
- `core/hardware.py` — trimmed per F2.
- `ingestion/documents/sections.py` — **untouched**; `reporting/` may still want the tail.

**Kept unchanged**: `catalog.py`, `fetcher.py`, `nifty50.py`, `notify.py`, `runlog.py`,
`documents/*` except a `page_spans` widening, `graph/state.py`, `graph/checkpoint.py`, all of
`reporting/` except the engine's one call site staying identical.

---

## C. Decisions needed before Phase 3

### C.1 — Years of history? **Affects cost by 3×**

The catalogue defaults to `annual_reports=1, concall_years=1`. Multi-year facts are what enable
trend claims ("guided to 15% margins three years running, missed twice") — plausibly the highest
value output in the design. Three years triples both Docling and SLM cost.
**Recommendation**: ship Phase 1-5 at 1 year; add `--years 3` as a separate follow-on run once
per-ticker cost is measured, not estimated.

### C.2 — Extraction model: `qwen-3.8-27b` or `gemma-4-31b`?

`GROWNXT_LLM_MODEL` defaults to `qwen-3.8-27b`. Schema-constrained extraction with a mechanical
quote-verbatim check is a forgiving task, so the default is probably fine.
**Recommendation**: bake off on 10 fixture sections at the start of Phase 4, decide on measured
discard rate, do not block Phases 1-3 on it.

### C.3 — `esg_brsr`: keep at weight 0.4, or drop?

29 pages and ~170k chars on ADANIENT, heavily templated, but material for the utilities and
Adani/Tata cohort. **Recommendation**: keep at 0.4; the boilerplate suppressor in Phase 6 is the
right tool for the template text, not a blanket drop.

### C.4 — Does `--fast-tables` survive?

With narrative spans running `do_table_structure=False`, the flag only affects the four note
classes. **Recommendation**: keep the flag and its `extract_version` suffix — the contract is
tested and the note tables are exactly where accurate mode matters.

---

## D. Target module design

### D.1 — File map, with responsibility and line budget

| Module | Budget | Responsibility | Must not |
| :--- | ---: | :--- | :--- |
| `qualitative/taxonomy.py` | 180 | `SECTION_CLASSES`, `FACT_TYPES`, keep/weight policy, `QUALITATIVE_VERSION` | contain any I/O or regex matching logic |
| `qualitative/prefilter.py` | 200 | pre-Docling `qualitative_page_spans()` | ever return a span set on an uncertain document (return `None`) |
| `qualitative/sectionmap.py` | 260 | heading tree → `list[SectionSpan]`, write `sectionmap.json` | call a model |
| `qualitative/segments.py` | 240 | transcript `SpeakerTurn`, presentation `Slide` | call a model |
| `qualitative/ledger.py` | 280 | `QualitativeFact`, `FactSet`, JSONL store, read-through resolver | call a model or the network |
| `qualitative/extractor.py` | 320 | the SLM pass; schema prompt, verbatim validation, caching | trust any number it reads |
| `qualitative/lexical.py` | 220 | BM25 index build + query, `lexical.npz` | import sqlite3, or any new dependency |
| `qualitative/query.py` | 180 | `LedgerQuery` → filter → rank → facts | know about pillars |
| `qualitative/routing.py` | 150 | the 6 pillars as `LedgerQuery`s, adaptive re-routing | know about BM25 |
| `qualitative/store.py` | 240 | `TickerState` / `DocumentIndexState`, `state.json` I/O | know about facts' content |
| `qualitative/boilerplate.py` | 160 | cross-ticker shingle suppression | be required for a single-ticker run |
| **Total new** | **~2,430** | against 3,192 deleted | |

### D.2 — Clean-code rules, as acceptance criteria

1. **No module both calls a model and decides policy.** `extractor.py` calls the SLM;
   `taxonomy.py` decides what to ask it about. A reviewer can read the policy without reading a
   prompt.
2. **`quote` verbatim-validation is in exactly one function**, and a failure increments a
   counter. Silent discards are the failure mode that makes a zero-hallucination claim untrue.
3. **One definition of "current"** — `document_reason` keeps that role, gains one comparison.
   Two copies of that predicate is how the inversion bug happened; do not add a second.
4. **No numbers from PDFs reach a report figure.** `metric_hint` is `dict | None` and no code
   path passes it to `reporting/fmt.py`. Enforced by a test, not a convention.
5. Google style, full type annotations, `console_utf8()` on every new CLI path.

### D.3 — The two new stage signatures

```python
def map_sections(ctx: StageContext, entry: CatalogEntry,
                 document: ExtractedDocument) -> SectionMapResult
def build_ledger(ctx: StageContext, entry: CatalogEntry,
                 sectionmap: SectionMap) -> LedgerResult
```

Both follow the existing convention exactly: explicit `StageContext`, failure in the result
object rather than raised, no knowledge of `TickerRun` or `RunManifest`.

`StageContext` field changes: `page_filter` → `qualitative_filter: bool`; add
`ledger_model: str`, `ledger_concurrency: int`; `indexer: QdrantVectorIndexer` →
`store: LedgerStore`. `wanted_extract_version()` folds `+qual-pages` in the same fixed order as
`+fast-tables`.

`BatchGraphState` gains nothing — `fact_count` rides in the existing per-ticker result dict as an
integer, per that module's identifiers-and-counts-never-payloads rule.

---

## E. Implementation phases

### Phase 0 — Branch and parity gate ✅ **done 2026-09-10**

Branch `feat/vectorless-qualitative-rag` cut from `feat/nifty50-resumable-graph`; all 68
uncommitted files carried over intact.

The gate is `scripts/parity_gate.py`, not a command list — the list could not tell a
pre-existing failure from a new one, and this tree has four of the former (F11).

```powershell
python scripts/parity_gate.py            # full: imports, lint, 4 suites (~2 min)
python scripts/parity_gate.py --quick    # imports and lint only (~15 s)
python scripts/parity_gate.py --suite scripts/verify_nifty50_embeddings.py
```

It sweeps package imports (F5) **and script imports separately** (F10), asserts the two
already-dead scripts are still dead *with the recorded error* — so their Phase 5 deletion
cannot silently not happen — then lints and runs the suites.

**Baseline: 19 of 23.** Every later phase holds at 19/23 or better. See F11 for what the four
failures are and which one to watch.

### Phase 1 — Rename `indexer` → `store`, no behaviour change (half day)

Pure mechanical rename across the 34 sites in F4, plus `ingestion/__init__.py`. `LedgerStore` is
at this point a thin subclass of `QdrantVectorIndexer` exposing `load_ticker_state`,
`save_ticker_state`, `namespace_for`. **Nothing else changes.** One reviewable commit that makes
every later diff readable.

*Gate*: full parity suite green, byte-identical `state.json` on a `--dry-run`.

### Phase 2 — Taxonomy, prefilter, section map (1.5 days) — *the polarity fix*

`taxonomy.py`, `prefilter.py`, `sectionmap.py`, `page_spans` widening in
`Extractor._cache_key`/`run()`, `map_sections` stage replacing `chunk_one`.
`--qualitative-filter` CLI flag; `page_filter` mapped to it per F8.

**Ships value on its own**: a per-ticker document map, correct pages ingested, ~70% less
TableFormer work, and no model involved anywhere.

*Gate*: parity suite, plus new `check_prefilter_polarity` and `check_auditor_certificate_guard`.

### Phase 3 — Speaker turns and slides (1 day)

`segments.py`. Transcript prepared-remarks/Q&A split with analyst firm attribution; presentation
slide units. Wired into `map_sections` as the non-annual-report path.

*Gate*: `check_speaker_turns` on a real transcript from `output/ADANIENT/`.

### Phase 4 — The Evidence Ledger (2 days) — *the expensive phase*

`ledger.py`, `extractor.py`, `store.py` proper, `build_ledger` stage,
`qualitative_version`/`ledger_fact_count` into `DocumentIndexState`,
`REASON_QUALITATIVE_VERSION` into `document_reason`. C.2's model bake-off happens here.

*Gate*: `check_quote_verbatim`, `check_no_numeric_authority`, and the load-bearing one —
`check_qualitative_version_diff`: a `QUALITATIVE_VERSION` bump re-runs the SLM pass and
**does not** re-run Docling (F6).

### Phase 5 — Lexical index, query, routing, cutover (1.5 days)

`lexical.py`, `query.py`, `routing.py`. Rewire `rag/pipeline.py`'s retrieve node, drop
`rerank_evidence` from the topology, adapt `synthesizer._build_evidence_context`.
`extract_ticker_findings` signature unchanged (F3).

Rewire `scripts/evaluate_full_pipeline.py` onto the new stages **before** the deletions (F10).

**Then delete**: `indexer.py`, `chunker.py`, `retriever.py`, `reranker.py`, `probes.py`,
`probe_vectors.json`, `verify_chunker.py`, and the two already-dead scripts
`ingest_documents.py` and `verify_ingestion.py` (F10) — which also retires CLAUDE.md's
ImportError note; prune `ingestion/__init__.py`;
drop `sentence-transformers` and `qdrant-client` from `requirements.txt`/`.lock`; strip the
Arctic layer from the Dockerfile; trim `core/hardware.py` per F2 (**keep `resolve_workers`**);
remove `QDRANT_*` from `.env.example`, `README.md` and the CLAUDE.md variable table.

*Gate*: full suite, plus `python scripts/generate_report.py WIPRO` compiling a PDF whose
findings section is populated — the end-to-end proof that F3's seam held.

### Phase 6 — Boilerplate suppression (half day)

`boilerplate.py`, `output/_nifty50/boilerplate.json`. Pure token-cost optimisation; correctly
a no-op on a cold single-ticker run.

### Phase 7 — Docs and live validation (half day)

Rewrite the CLAUDE.md architecture section, the gotchas that reference Qdrant/embeddings, and
the README. Then `python -m scripts.embed_nifty50 --limit 3` twice, asserting the second run is
a no-op, and one `--live WIPRO`.

**Total: ~8 working days.** Phases 1-2 are independently shippable; the point of no return is
the deletion block in Phase 5.

---

## F. Verification plan

`scripts/verify_qualitative_rag.py`, on `scripts/checks.py`, offline by default:

| Check | Guards |
| :--- | :--- |
| `check_taxonomy_closure` | every class has keep/weight/fact_types; `QUALITATIVE_VERSION` moves when any does |
| `check_prefilter_polarity` | ADANIENT fixture: **includes** p166 (MD&A), **excludes** p300 (notes) |
| `check_prefilter_fails_open` | 40-page / anchorless / unreadable → `None` |
| `check_auditor_certificate_guard` | p191 governance certificate does not anchor KAM |
| `check_quote_verbatim` | non-substring quote is discarded **and counted** |
| `check_no_numeric_authority` | no path writes `metric_hint` into a report figure |
| `check_qualitative_version_diff` | version bump → SLM re-run, **no** Docling re-run |
| `check_speaker_turns` | prepared/Q&A split with analyst firms |
| `check_bm25_determinism` | same query + ledger → same order, cross-platform |
| `--live SYM` | one real ticker, twice, second run a no-op |

Retained in `verify_nifty50_embeddings.py` per F7: Layer 1 diff, `document_reason` inversion
guard, catalogue TTL, `pid_alive`, `_check_launch_handoff`, hardware resolution,
`resolve_workers` capping, extractor-version/table-mode contract.

---

## G. Risks

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| SLM cost at 50 tickers exceeds budget | **High** | Measure on 3 tickers at end of Phase 4, before Phase 5's deletions. The fallback — stop at Phase 3 and BM25 over section text with no fact extraction — is still fully vectorless and still fixes the polarity. Keep that exit open until Phase 4's numbers are in. |
| Heading regexes miss a section on an issuer whose report is laid out differently | **High** | Unclassified → `business_overview` at `confidence=0.3`, biased to keep. Run `map_sections` over all 50 and eyeball the class histogram before Phase 4 spends any tokens. |
| Deleting `indexer.py` loses the `upload_collection` and last-batch-wait optimisations | Medium | They are documented in CLAUDE.md and recoverable from git. Nothing in the new path needs them. |
| `reporting/engine.py` breaks despite F3 | Medium | Phase 5's gate is a compiled WIPRO PDF, not a unit test. |
| `transformers` version conflict on removing `sentence-transformers` | Low | F1 — removal *relaxes* the constraint. Re-lock and confirm `docling` still imports. |
| Windows `cp1252` crash on `₹` in new CLI output | Low | `console_utf8()` in every new entry point; it is a D.2 acceptance criterion. |

---

## H. Checklist

- [x] Phase 0 — branch cut, `scripts/parity_gate.py` written, baseline 19/23 recorded (F11)
- [ ] Phase 1 — `indexer` → `store` rename, one commit, suite green
- [ ] Phase 2 — taxonomy + prefilter + section map; polarity tests green
- [ ] Phase 3 — speaker turns + slides
- [ ] Phase 4 — ledger + SLM pass; **cost measured on 3 tickers**; C.2 decided
- [ ] Phase 5 — lexical + query + routing; cutover; `evaluate_full_pipeline.py` rewired (F10);
      **deletion block**; WIPRO PDF compiles
- [ ] Phase 6 — boilerplate suppression
- [ ] Phase 7 — CLAUDE.md, README, `.env.example`; two `--limit 3` runs; one `--live`
- [ ] Spec §5 and §9 amended to match this plan's replacement scope
- [ ] Spec §1 and §7 amended per F1 (torch stays) and F2 (worker cap stays)
