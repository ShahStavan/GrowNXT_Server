# Spec: Vectorless Qualitative RAG for Nifty 50 Fundamental Research

- **Status**: Draft, for review
- **Author**: Claude (via /specs), for @jenish.gajera
- **Date**: 2026-09-10
- **Target branch**: new `feat/vectorless-qualitative-rag` off `feat/nifty50-resumable-graph`
- **Extends**: `.claude/specs/nifty50-embedding-pipeline.md` (catalogue, two-layer change
  detection, run manifest), `.claude/specs/nifty50-resumable-langgraph-orchestration.md`
  (checkpointed stage graph). Both contracts stay intact.
- **Supersedes**: the retrieval half of the pipeline — `ingestion/indexer.py` as the *primary*
  retrieval structure, and `ingestion/documents/sections.py`'s page-selection *polarity*.

---

## 1. Goal

Produce, for each Nifty 50 constituent, the **qualitative** evidence a fundamental analyst needs —
what management says it is doing, why margins move, where capital is going, what the risks are —
and make it retrievable **without a vector index**.

Three facts about this problem, taken together, dissolve it:

1. **The quantitative half is already solved and is better solved elsewhere.** Fourteen REST
   endpoints (`reporting/client.py:30`) serve income, balance sheet, cash flow, quarterly,
   DuPont, solvency, liquidity, capital efficiency and CAGR as clean JSON. Nothing in an annual
   report's financial statements needs to be re-derived from a PDF, and re-deriving it is how
   numeric hallucinations get in.
2. **Qualitative content is *structurally addressable*, not semantically scattered.** An analyst
   does not fuzzy-search an annual report. They go to the Management Discussion & Analysis. Then
   the Board's Report. Then the Key Audit Matters. The retrieval key is a **named section**, and
   a section has a page range. That is a lookup, not a nearest-neighbour search.
3. **The question set is closed and already written down.** `ingestion/rag/probes.py` enumerates
   six research pillars. When the questions are known at ingest time, the answers can be
   *precomputed* at ingest time. This is the actual unlock: retrieval stops being a search
   problem and becomes a **filter over a typed record store**.

> **Design principle.** Embeddings exist to find text when you do not know where to look. For a
> corpus of 50 known issuers, 3 known document classes, ~15 known section types and 6 known
> question pillars, you always know where to look. Paying for an embedding model, a GPU, a vector
> database and an ANN index to answer "what did management say about margins" — when the answer
> is on the two pages headed *Management Discussion and Analysis → Operating Margins* — is
> paying for generality this problem does not have.

### Success criteria

- Zero embedding model, zero model weights and zero Qdrant **in the query path**. Docling's
  torch dependency in the *ingest* path is unchanged — see §5, which corrects an earlier
  "no torch" claim in this spec.
- Financial-statement pages are **excluded** from ingestion; the four note classes the REST API
  does not carry are **retained** by name (§4.2).
- Every sentence in a generated finding cites a `fact_id` resolvable to a verbatim quote and a
  page number in a named source document.
- A steady-state re-run of all 50 constituents does no Docling work and no SLM work.
- Per-ticker cold ingest cost drops by the numbers in §7, measured, not estimated.

---

## 2. Background — what the pipeline does today, audited

Line references are to `HEAD` of `feat/nifty50-resumable-graph`.

### 2.1 The page filter keeps precisely the half we must discard

`ingestion/documents/sections.py` is a well-built module solving the **opposite** problem. Its
docstring is explicit: the financial content of an annual report is a contiguous tail, so it finds
where that tail starts and converts *from there to the end*. `financial_page_range()` returns
`(start, total)`.

Measured on the real extraction on disk, `output/ADANIENT/extracted/annual_report_FY2026.json`
(396 source pages, 2,106,468 chars, 11,260 blocks, 626 tables):

| Page span | Content | Chars | Tables | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| 1–153 | Corporate overview, portfolio, strategic review, ESG narrative | ~620k | ~90 | **Keep** — highest-value strategy content |
| 154–157 | Board's Report | ~25k | ~5 | **Keep** |
| 158–172 | Management Discussion & Analysis | ~95k | ~20 | **Keep** — the single densest source |
| 173–192 | Corporate Governance Report | ~120k | ~50 | **Mostly drop** — statutory boilerplate |
| 193–221 | Business Responsibility & Sustainability Report | ~170k | ~60 | **Keep, downweighted** |
| 222–229 | Independent Auditor's Report incl. Key Audit Matters | ~50k | ~20 | **Keep** — not in the API |
| 230–396 | Financial statements and notes | ~1,030k | ~438 | **Drop**, except 4 named note classes |

`financial_page_range()` on this document anchors at **page 222** and converts **222–396**. It
therefore keeps ~1.07M chars that duplicate the REST API, and discards ~1.03M chars of exactly the
management commentary this project now needs. The module is not wrong; it was built for a report
that publishes numbers. Its polarity has to invert.

### 2.2 Retrieval is a vector index that never needed to be one

`ingestion/indexer.py` (1,617 lines) runs Snowflake Arctic `snowflake-arctic-embed-m-v1.5` at 768
dimensions into a per-ticker Qdrant collection, mirrored to `output/<TICKER>/vectors.npz`. It
carries — correctly, and at real engineering cost — a GPU worker cap, an fp16 path, a
`upload_collection` fast path that avoids materialising 2.4M Python floats per annual report, and
a last-batch-only wait optimisation.

All of that machinery exists to serve `ParallelVectorRetriever.retrieve_all_probes_parallel()`
(`ingestion/rag/retriever.py:264`), which issues **15 fixed queries** — five pillars × three
phrasings — that are hardcoded in `probes.py` and cached to `probe_vectors.json`. Fifteen static
queries per ticker. The index is a general-purpose ANN structure answering a fixed, tiny,
known-in-advance query set.

### 2.3 Chunking is document-shaped, not analyst-shaped

`chunk_document()` (`ingestion/chunker.py:344`) produces 800-char `SentenceSplitter` windows over
whatever blocks arrive, plus a table chunk per table and a figure chunk per figure. For ADANIENT
that is 7.7 MB of chunk JSON. A chunk carries `section` and `heading_path`, but nothing classifies
those strings — `sections` in the extraction includes `'2.'`, `'~7 million 3'` and
`'1 In FY 2025-26'` alongside `'Corporate Overview'`. There is no taxonomy, so there is no way to
say "give me the MD&A".

### 2.4 What is already right and must be preserved

- **Catalogue and identity** (`ingestion/catalog.py`): period-derived `doc_id`s, concall
  disambiguation, `M&M` percent-encoding. Untouched.
- **Two-layer change detection** (`ingestion/stages.py:337`, `indexer.py:589`): the version-bump
  and collection-change reasons. Extended in §4.7, not replaced.
- **Stage decomposition** (`ingestion/stages.py`): explicit `StageContext`, results-not-exceptions.
  New stages follow the same convention.
- **Graph state discipline** (`ingestion/graph/state.py`): identifiers and counts, never payloads.
- **Extractor cache keying** (`extract.py:312`): `page_range` already participates in
  `_cache_key`, and `extract_version()` already carries every text-changing setting. Both are
  exactly the hooks this design needs.

---

## 3. Architecture

Four layers. Each is cheaper than the one below it and answers most queries before the next one
runs.

```
                     QUERY: "why are margins compressing at TCS?"
                                        │
  ┌─────────────────────────────────────▼──────────────────────────────────────┐
  │ L0  ROUTE — deterministic, no model, microseconds                          │
  │     ticker → document manifest → doc classes that can answer this pillar   │
  │     ingestion/catalog.py (exists) + qualitative/routing.py (new)           │
  └─────────────────────────────────────┬──────────────────────────────────────┘
                                        │  doc_ids: [AR FY2026, TR 2026_07, ...]
  ┌─────────────────────────────────────▼──────────────────────────────────────┐
  │ L1  SECTION MAP — structural index, a lookup not a search                  │
  │     doc_id → [SectionSpan{class, pages, heading_trail, n_chars}]           │
  │     output/<TICKER>/qualitative/sectionmap.json                            │
  └─────────────────────────────────────┬──────────────────────────────────────┘
                                        │  scope: MD&A p158-172, concall Q&A turns 14-31
  ┌─────────────────────────────────────▼──────────────────────────────────────┐
  │ L2  EVIDENCE LEDGER — precomputed typed claims (the retrieval unit)        │
  │     filter(fact_type, section_class, period, subject) → BM25 rank          │
  │     output/<TICKER>/qualitative/facts.jsonl + lexical.npz                  │
  └─────────────────────────────────────┬──────────────────────────────────────┘
                                        │  facts + verbatim quotes + page cites
  ┌─────────────────────────────────────▼──────────────────────────────────────┐
  │ L3  READ-THROUGH — escape hatch when the ledger is thin                    │
  │     (doc_id, page span) → verbatim blocks from extracted/<doc_id>.json     │
  └─────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                    SYNTHESIS — reporting/, ingestion/rag/synthesizer.py
                    every sentence cites a fact_id (unchanged contract)
```

### 3.1 Why this is "vectorless" and not merely "keyword search"

The common failure of dropping embeddings is falling back to BM25 over raw chunks, which is worse
at everything. This design does not do that. Three mechanisms replace semantic similarity:

- **Structural addressing (L1)** replaces "find text about strategy" with "open the section named
  *Strategic Review*". Recall comes from the taxonomy, not from term overlap.
- **Precomputed normalisation (L2)** replaces query-time paraphrase matching with ingest-time
  paraphrase resolution. Management writes "we expect operating leverage to play out through
  H2"; the ledger records `fact_type=guidance, subject=margin, stance=forward_looking` with the
  verbatim quote attached. The query filters on the *type*, so the phrasing never has to match.
- **Lexical rank is only a tie-break (L2)** over an already-tiny, already-correct candidate set —
  typically 20–200 facts, not 40,000 chunks. BM25 is excellent at that scale and terrible at the
  other.

---

## 4. Component design

### 4.1 New package layout

```
ingestion/qualitative/
├── __init__.py          # public surface: build_ledger, query, SECTION_CLASSES, FACT_TYPES
├── taxonomy.py          # SECTION_CLASSES, FACT_TYPES, KEEP/DROP policy, QUALITATIVE_VERSION
├── prefilter.py         # pre-Docling page selection (replaces sections.py polarity)
├── sectionmap.py        # post-Docling heading-tree → SectionSpan classification
├── segments.py          # transcript speaker turns; presentation slide units
├── boilerplate.py       # cross-ticker shingle suppression
├── ledger.py            # QualitativeFact, FactSet, JSONL store, read-through resolver
├── extractor.py         # SLM fact-extraction pass (batched, cached, deterministic prompts)
├── lexical.py           # pure-Python BM25 over the ledger; no SQL, no new dependency
└── routing.py           # pillar → (doc_type, section_class, fact_type) routing table
```

`ingestion/documents/sections.py` is **retained unchanged** — the report generator in
`reporting/` may still want the financial tail — and `prefilter.py` becomes the batch path's
selector. Two modules with opposite polarities, each named for what it keeps.

### 4.2 `taxonomy.py` — the canonical section classes

The one table the whole design hangs on. `keep` drives page selection; `weight` drives ranking;
`fact_types` scopes the SLM prompt so the extractor is never asked open-ended questions.

| `section_class` | Keep | Weight | Rationale |
| :--- | :--- | ---: | :--- |
| `chairman_letter` | ✅ | 0.9 | Strategy in management's own framing. Short, dense. |
| `mdna` | ✅ | 1.0 | Highest signal per page in the entire filing. |
| `boards_report` | ✅ | 0.8 | State of affairs, material changes, subsidiary moves. |
| `business_overview` | ✅ | 0.7 | Segments, products, geographies, moat claims. |
| `segment_review` | ✅ | 0.9 | Per-segment drivers — the API gives segment *numbers*, not *why*. |
| `risk_management` | ✅ | 0.9 | Named risks with mitigation; maps to the risk pillar. |
| `esg_brsr` | ✅ | 0.4 | Material for some sectors; heavily templated. Downweighted, not dropped. |
| `kam` | ✅ | 1.0 | Key Audit Matters. Not in the API. Small, and the highest-value few pages in the back half. |
| `notes_contingent` | ✅ | 0.9 | Contingent liabilities and litigation. Not in the API. |
| `notes_rpt` | ✅ | 0.8 | Related-party transactions. Not in the API. Governance red flags live here. |
| `notes_segment` | ✅ | 0.7 | Segment note commentary and accounting policy for segments. |
| `corp_governance` | ⚠️ | 0.2 | Statutory template. Keep only board-composition changes and RPT policy; drop the rest. |
| `notice_agm` | ❌ | — | Pure procedure. |
| `financial_statements` | ❌ | — | **Redundant with `reporting/client.py`.** Excluded by policy, not by cost. |
| `notes_other` | ❌ | — | Accounting-policy notes covered by the API's derived ratios. |
| `boilerplate` | ❌ | — | Disclaimers, forward-looking-statement notices, awards pages, credits. |

`QUALITATIVE_VERSION` is a single string folded from the taxonomy hash, the fact schema version
and the extractor prompt version. It participates in change detection exactly as
`CHUNKER_VERSION` does today (§4.7).

### 4.3 `prefilter.py` — pre-Docling page selection

Same cheap-pypdf-pass technique as `sections.py`, same fail-safe asymmetry, inverted target.

```python
def qualitative_page_spans(pdf, min_pages=60) -> list[tuple[int, int]] | None
```

Returns **a list of contiguous spans**, because the qualitative content is *not* contiguous: the
narrative front plus the KAM block plus a handful of note pages. Docling is invoked once per span
and the block streams are concatenated in page order; `Extractor._cache_key` already takes a
`page_range`, so this needs a `page_spans` tuple in the key rather than a new mechanism.

Algorithm:
1. `page_texts()` (reuse `sections.py`'s, promoted to a shared helper).
2. Locate `fin_start` with the existing `ANCHORS`, including the `AUDITOR_REPORT` "report"-word
   guard — that hard-won false-positive fix (governance certificate on page 191) applies
   identically here and must not be re-derived.
3. Locate `kam_end` — the first page after `fin_start` matching a balance-sheet or
   statement-of-profit-and-loss heading. The KAM block is `[fin_start, kam_end)`.
4. Scan `[kam_end, total]` for `notes_contingent` / `notes_rpt` / `notes_segment` anchors; emit
   each hit page ±1 as a span, then merge overlaps.
5. Drop `notice_agm` from the front by locating the first non-notice anchor.
6. **Return `None` for every uncertain case** — under `min_pages`, no anchors, scanned PDF,
   fewer than half the pages with extractable text. `None` means convert everything, and the
   post-Docling classifier in §4.4 then does the filtering on the block tree instead. A dropped
   page is unsearchable forever; a spare one costs seconds. Unchanged asymmetry, and this is why
   the classifier must be able to stand alone.

Measured on ADANIENT FY2026 this yields roughly `[(2, 221), (222, 229), (234, 238), (259, 261),
(269, 270), (335, 341), (345, 347)]` — **~240 of 396 pages, and ~188 of 626 tables**. The page
saving is modest; the *table* saving is 70%, and TableFormer is the expensive part.

### 4.4 `sectionmap.py` — the navigable index

Post-Docling, walk `ExtractedDocument.blocks` and classify each `KIND_HEADING` into a
`section_class` by regex against the taxonomy, inheriting down the heading tree. Emit:

```python
@dataclass
class SectionSpan:
    section_class: str  # from SECTION_CLASSES
    heading: str  # verbatim heading text
    heading_trail: str  # "Statutory Reports > Board's Report > Risk Management"
    page_start: int
    page_end: int
    block_start: int  # index into ExtractedDocument.blocks — the read-through key
    block_end: int
    n_chars: int
    keep: bool
    weight: float
    confidence: float  # regex hit = 1.0; inherited = 0.7; LLM-resolved = 0.5
```

Written to `output/<TICKER>/qualitative/sectionmap.json`. Unclassified headings inherit the
enclosing span's class; a top-level heading matching nothing becomes `business_overview` with
`confidence=0.3` — biased toward keeping, consistent with §4.3.

**This file alone is a usable deliverable.** It is the table of contents an analyst navigates, it
is what an agent lists before drilling in, and it renders as a per-ticker document map with no
model involved.

### 4.5 `segments.py` — transcripts and presentations

Concall transcripts have the strongest structure in the corpus and the current pipeline throws it
away into 800-char windows.

**Transcript → `SpeakerTurn`**: `speaker`, `role` (`management` | `analyst` | `moderator`),
`firm` (analysts), `phase` (`prepared_remarks` | `qa`), `turn_index`, `page`, `text`. Detection is
a `^\s*([A-Z][A-Za-z.\- ]{2,40}):` pattern plus a moderator-phrase list; the prepared/Q&A boundary
is the first moderator turn containing a question-invitation phrase.

Two things this buys that no chunker can:
- **Analyst questions are a separate evidence class.** They are the market's stated concerns, and
  they are the best available prior on what the next quarter's controversy will be. Stored as
  `fact_type=analyst_concern` with the asking firm attached.
- **Attribution.** "The CFO guided to 15% EBITDA margin" is a materially different claim from an
  analyst positing it. The ledger records `speaker` and `role` on every transcript fact.

**Presentation → `Slide`**: title, bullets, page. Numbers on slides are management's *framing* and
are recorded as `metric_hint` only — never as a source of truth. The API wins every numeric
conflict, always, and a conflict is itself worth surfacing (§4.9).

### 4.6 `ledger.py` + `extractor.py` — the Evidence Ledger

The retrieval unit. One record per claim.

```python
@dataclass
class QualitativeFact:
    fact_id: str  # sha256(ticker|doc_id|block_start|statement)[:16]
    ticker: str
    doc_id: str
    doc_type: str  # annual_report | transcript | presentation
    period: str  # "FY2026" | "2026Q1" — normalised, sortable
    section_class: str
    fact_type: str  # see FACT_TYPES below
    subject: str  # segment / geography / product / "consolidated"
    statement: str  # 1-2 sentence normalised claim, model-written
    quote: str  # verbatim source text, <= 300 chars, model-copied
    page: int
    block_start: int  # read-through key into extracted/<doc_id>.json
    block_end: int
    speaker: str = ""  # transcripts
    speaker_role: str = ""
    stance: str = "historical"  # forward_looking | historical
    horizon: str = ""  # "FY2027", "H2", "3-5 years"
    metric_hint: dict | None = None  # {name, value, unit} — framing, never truth
    confidence: float = 0.0
```

`FACT_TYPES`: `guidance`, `target`, `growth_driver`, `headwind`, `margin_commentary`,
`pricing_power`, `cost_pressure`, `capex_plan`, `capital_allocation`, `segment_commentary`,
`demand_commentary`, `order_book`, `management_change`, `regulatory`, `litigation`,
`related_party`, `esg_commitment`, `audit_matter`, `risk_factor`, `analyst_concern`.

**The extraction pass.** For each kept `SectionSpan` (or `SpeakerTurn`, or `Slide`), one SLM call
against `core/llm_config.py`'s hosted endpoint with a **closed-form JSON schema** and the
`fact_types` valid for that `section_class`. Three rules make this safe:

1. **`quote` must be a verbatim substring of the input.** Validated in Python after the call;
   a fact whose quote does not appear in its source block is **discarded**, not repaired. This
   is the zero-hallucination guarantee, enforced mechanically rather than by prompt.
2. **No numbers are trusted.** `metric_hint` is advisory. Anything the report presents as a
   financial figure comes from `reporting/client.py`.
3. **`clean_thinking_tokens` on every response**, per the existing project rule.

Output `output/<TICKER>/qualitative/facts.jsonl`, one JSON object per line, append-only per
document, with `facts.meta.json` recording `QUALITATIVE_VERSION`, per-doc fact counts and the
prompt hash.

Expect 40–120 facts per annual report, 25–60 per transcript, 15–40 per presentation. **Roughly
150–250 facts per ticker per year** — an object small enough to hold entirely in memory for all
50 constituents at once (~10k facts, a few MB).

### 4.7 Change detection — extending, not replacing

`DocumentIndexState` (`indexer.py:240`) gains two fields alongside `chunker_version` and
`extract_version`:

```python
qualitative_version: str = ""  # taxonomy + fact schema + prompt hash
ledger_fact_count: int = 0
```

`should_index_document()`'s reason set gains `REASON_QUALITATIVE_VERSION`. The Layer 1 diff in
`stages.py:337` treats a `qualitative_version` mismatch exactly as it treats a
`CHUNKER_VERSION` bump: re-run the SLM pass, reuse the Docling extraction on disk if
`extract_version` still matches. This is the important economy — a taxonomy or prompt change
costs SLM calls, never a re-extraction.

> **Guard.** `should_index_document`'s final comparison was inverted once before, and
> `scripts/verify_nifty50_embeddings.py` exists partly to catch that. The new field goes into the
> same comparison and the same test. Do not add a parallel code path.

### 4.8 `lexical.py` — BM25 without a database

Ranking within an already-filtered candidate set. Pure Python + NumPy, no new dependency, no SQL —
`CLAUDE.md`'s "no database engine" rule stands.

- Per-ticker inverted index over `statement + quote + heading_trail`, built at ingest.
- Persisted as `output/<TICKER>/qualitative/lexical.npz` (`indptr`, `indices`, `tf` int32 CSR) +
  `vocab.json`. Same file-backed shape as the existing `vectors.npz` mirror.
- Query-time: `k1=1.2, b=0.75`, plus `weight` from the section taxonomy and a recency multiplier
  on `period`.
- Tokeniser: lowercase, Indian-numeral-aware (`₹`, `crore`, `lakh`, `bps`), with a financial
  synonym expansion table (`ebitda|operating profit`, `capex|capital expenditure`,
  `guidance|outlook|expects`) applied at **query** time only, so the index stays literal.

At 150–250 facts per ticker this is sub-millisecond and needs no ANN structure of any kind.

### 4.9 `routing.py` — pillars become queries, not embeddings

`probes.py`'s six pillars are re-expressed as declarative ledger queries. No embedding, no
`probe_vectors.json`, no `ParallelVectorRetriever`.

```python
PILLAR_QUERIES = {
  PILLAR_MARGIN_COST: LedgerQuery(
      fact_types=["margin_commentary", "cost_pressure", "pricing_power", "headwind"],
      section_classes=["mdna", "segment_review", "boards_report"],
      doc_types=["transcript", "annual_report"],
      prefer_stance="forward_looking",
      terms="ebitda margin cost inflation pricing realisation operating leverage",
      top_k=12,
  ),
  ...
}
```

`probes.py`'s `build_adaptive_probes()` adaptive-routing behaviour — re-target to the annual
report's MD&A when a ticker has no transcript — is preserved verbatim in shape, now operating on
`section_class` instead of `doc_type` strings, which makes it strictly more precise.

**Cross-check surface.** Because the ledger carries `metric_hint` and the API carries truth, a
cheap deterministic comparator can flag where management's framing and the reported numbers
diverge. That is a genuinely new analytical output this architecture makes almost free, and it is
the kind of thing a sell-side note leads with. Out of scope for phase 1; noted so the schema
supports it.

### 4.10 `boilerplate.py` — cross-ticker suppression

Indian annual reports share enormous volumes of statutory template text. Suppressing it before the
SLM pass is where the token budget is actually won.

64-bit rolling hash over 8-token shingles; a shingle appearing in ≥ 5 distinct tickers is
template. Persisted as `output/_nifty50/boilerplate.json` (counts, not a bloom filter — 50 tickers
does not need probabilistic compression, and an exact structure stays debuggable). A block whose
shingles are ≥ 70% template is skipped. Bootstrapped on the first full run and refined thereafter;
a cold single-ticker run simply has no suppression, which is correct.

---

## 5. What happens to the existing vector stack

> **Amended 2026-09-10.** This section originally proposed keeping the vector stack behind a
> `GROWNXT_RETRIEVAL_MODE` flag. That is superseded: the decision is **full replacement**, with
> no `hybrid` and no `vector` mode. See
> [`.claude/plans/vectorless-qualitative-rag.md`](../plans/vectorless-qualitative-rag.md) for the
> phased removal and its call-site audit.

**Deleted** — 3,192 lines: `ingestion/indexer.py` (1,617), `ingestion/chunker.py` (677),
`ingestion/rag/retriever.py` (511), `ingestion/rag/reranker.py` (200),
`ingestion/rag/probes.py` (187), and `scripts/verify_chunker.py`. With them go
`output/<T>/chunks/`, `vectors.npz`, `payloads.json`, `probe_vectors.json`, the `QDRANT_*`
environment variables, and the Dockerfile's pre-cached 400 MB Arctic layer.

**Two things that do *not* leave, contrary to an earlier draft of this spec:**

- **torch stays.** `docling-ibm-models==4.0.1` hard-requires `torch>=2.2.2` and `torchvision` —
  Docling's layout model and TableFormer *are* torch models. Removing the embedder removes
  `sentence-transformers` and `qdrant-client`, not torch. It does relax a real conflict:
  `sentence-transformers` demands `transformers>=5.0.0` while `docling-ibm-models` demands
  `<5.9.0`.
- **The GPU worker cap stays.** `resolve_workers()` caps `--workers` to 1 under 8 GB VRAM
  because each worker thread loads its own **layout and table** models onto the same card. That
  was never about the embedder. By the same argument `core/hardware.py` is trimmed rather than
  removed — `device`, `num_threads`, `idle_gpu` and `resolve_workers` all still serve Docling;
  only `embed_batch_size` and `embed_fp16` become dead.

The single interface that must survive byte-for-byte is
`extract_ticker_findings(ticker, company_name, force) -> ResearchDossier`
(`ingestion/rag/pipeline.py:491`), because `reporting/engine.py:134` is its only external
caller. `ThematicFinding` and `ResearchDossier` keep their schemas, so `findings.json` stays
compatible.

---

## 6. Pipeline integration

Two new stages in `ingestion/stages.py`, following the existing `StageContext` → result-object
convention, replacing `chunk_one` and `index_chunk_files` in the batch path:

```
catalogue → diff → download → extract → map_sections → build_ledger → mirror
                                            (new)         (new)
```

- `map_sections(ctx, extracted) -> SectionMapResult` — pure Python, no network, no model.
  Deterministic and instant, so it re-runs freely on a taxonomy change.
- `build_ledger(ctx, sectionmap) -> LedgerResult` — the SLM pass. The only stage with a network
  dependency after download, and the only one worth checkpointing mid-document.

`ingestion/graph/nodes.py` gains the two nodes; `BatchGraphState` gains nothing — it carries
`fact_count` as an integer, per the identifiers-and-counts-never-payloads rule that module
enforces.

`StageContext` gains `page_filter` → **`qualitative_filter: bool`** (the inverted policy),
`ledger_model: str`, and `ledger_concurrency: int`. `wanted_extract_version()` folds
`+qual-pages` in the same fixed-order way `+fast-tables` and `+fin-pages` are folded today.

**Extractor settings for the batch path**: `ocr=False, figures=False` (already the case),
`accurate_tables=False`, and — new — `do_table_structure=False` for spans classified as narrative.
Tables in the qualitative half are mostly layout artefacts; the four note classes that genuinely
need table structure get their own spans with it enabled.

---

## 7. Cost, measured against ADANIENT FY2026

| | Today | This design | Δ |
| :--- | ---: | ---: | ---: |
| Pages through Docling | 396 | ~240 | −39% |
| Tables through TableFormer | 626 | ~188 | −70% |
| TableFormer mode | accurate | fast / off for narrative | — |
| Chunk JSON on disk | 7.7 MB | ~0.4 MB (ledger + map) | −95% |
| Text embedded | 2.1 M chars | 0 | −100% |
| Embedding model | Arctic 768d on GPU | none | removed |
| Vector store | Qdrant, per-ticker collection | none | removed |
| torch / torchvision | Docling + Arctic | Docling only | **stays** (§5) |
| SLM calls (ingest) | 0 | ~60–90 per annual report | **new cost** |
| SLM calls (query) | 6 pillars | 6 pillars | unchanged |
| Query latency | ANN + rerank | dict filter + BM25 | ~10× faster |
| Retrieval precision | similarity over 40k chunks | filter over ~200 typed facts | qualitative |

The honest trade: **ingest-time SLM calls are a new, real cost**, roughly 60–90 per annual report
against a hosted endpoint. It is paid once per document per `QUALITATIVE_VERSION`, it replaces GPU
embedding time rather than adding to it, and it is what converts retrieval from a search problem
into a lookup. If that cost proves unacceptable at 50 tickers, §9 has the cheaper variant.

---

## 8. Verification

New `scripts/verify_qualitative_rag.py`, built on `scripts/checks.py`, offline by default:

1. `check_taxonomy_closure` — every `section_class` has a keep policy, a weight and a non-empty
   `fact_types` set; `QUALITATIVE_VERSION` changes when any of them changes.
2. `check_prefilter_polarity` — on a fixture derived from the real ADANIENT heading list,
   `qualitative_page_spans()` **includes** page 166 (MD&A) and **excludes** page 300 (notes).
   This is the regression that inverting `sections.py`'s polarity exists to prevent.
3. `check_prefilter_fails_open` — a 40-page document, an anchorless document and an unreadable
   file each return `None`.
4. `check_auditor_certificate_guard` — the page-191 governance-certificate string does not
   anchor the KAM span. Mirrors the existing `check_page_filter` guard; the false positive is the
   same one.
5. `check_quote_verbatim` — a synthetic model response whose `quote` is not a substring of its
   source block is discarded, and the discard is counted, not silent.
6. `check_no_numeric_authority` — no code path writes a `metric_hint` value into a report figure.
7. `check_qualitative_version_diff` — a `QUALITATIVE_VERSION` bump re-runs the ledger and
   **does not** re-run Docling when `extract_version` is unchanged. This is the economy in §4.7;
   without a test it silently regresses into a full re-extraction.
8. `check_speaker_turns` — a transcript fixture splits into prepared remarks and Q&A with
   analyst firms attached.
9. `check_bm25_determinism` — same query, same ledger, same order, across runs and platforms.
10. `--live SYM` — one real ticker end-to-end, twice, asserting the second run is a no-op.

`scripts/verify_nifty50_embeddings.py` stays green throughout; the vector path is not modified.

---

## 9. Phasing

> **Amended 2026-09-10.** The authoritative, call-site-audited phasing now lives in
> [`.claude/plans/vectorless-qualitative-rag.md`](../plans/vectorless-qualitative-rag.md) §E —
> eight phases over ~8 working days, ending in deletion rather than a flag. The table below is
> retained as the dependency ordering it was written to express.

| Phase | Deliverable | Depends on | Value if you stop here |
| :--- | :--- | :--- | :--- |
| **1** | `taxonomy.py`, `sectionmap.py`, `prefilter.py`, verify suite | nothing | Per-ticker document map; correct pages ingested; 70% less TableFormer. **Ship this first — it is the whole polarity fix and it needs no model.** |
| **2** | `segments.py` | 1 | Speaker-attributed transcript units; analyst-concern extraction becomes possible. |
| **3** | `ledger.py`, `extractor.py`, `lexical.py` | 1, 2 | The vectorless retrieval path exists end to end. |
| **4** | `routing.py`, `synthesizer.py` rewire, cutover and deletion | 3 | Default path is vectorless; the vector stack is gone. |
| **5** | `boilerplate.py`, metric cross-check | 4 | SLM token cost drops; framing-vs-reported divergence becomes a report section. |

**The cheaper variant, if phase 3's SLM cost is unacceptable**: stop at phase 2 and retrieve by
`section_class` + BM25 over section text directly, with no fact extraction. That is still fully
vectorless and still solves the polarity problem — it just returns paragraphs instead of typed
claims, and the synthesizer does more work per query. Phase 1 and 2 are worth doing regardless of
how phase 3 resolves, which is why they are ordered first.

---

## 10. Decisions taken, and the ones left open

**Taken, as stated above** — flagged so they are easy to reverse:

- Financial statements excluded **by policy**, not by cost, with four named note exceptions.
- ~~Vector stack retained behind a flag rather than deleted.~~ **Reversed 2026-09-10: full
  replacement, no flag.** See §5 and the plan's §B.
- BM25 in pure Python rather than SQLite FTS5, to honour the no-database rule literally.
- `sections.py` kept and a second, opposite-polarity module added, rather than parameterising one
  module with a direction flag. Two clearly-named modules beat one module with a polarity
  argument that will eventually be passed wrong.

**Open, and worth a decision before phase 3:**

1. **How many years back?** The catalogue defaults to one annual report and one concall year.
   Multi-year facts are what make trend claims possible ("management has guided to 15% margins
   for three consecutive years and missed twice") — arguably the highest-value output in the whole
   design. Three years of annual reports triples ingest cost.
2. **`esg_brsr` — keep at weight 0.4, or drop?** Sector-dependent. Heavy pages, mostly templated,
   but material for the Adani/Tata/utilities cohort.
3. **Which model for the fact-extraction pass?** `qwen-3.8-27b` is the configured default and is
   likely adequate for schema-constrained extraction, where the quote-verbatim check catches
   drift mechanically. Worth a bake-off against `gemma-4-31b` on ten fixture sections before
   committing 50 tickers to it.

---

## 11. `core/hardware.py` — why compute resolution exists at all

Relocated from that module's docstring by
[`code-style-refactor.md`](../plans/code-style-refactor.md) Phase 4.

**Each library defaults badly on its own, and two of them fail silently.**

- **Docling** (layout + TableFormer) reads `AcceleratorOptions`, whose `num_threads` is
  **4 on every machine** and whose `device` is `auto`. Four threads is a floor, not a ceiling,
  and `auto` silently means CPU whenever torch was installed from the CPU wheel index.
- **torch** defaults its intra-op thread pool to the *logical* core count, which oversubscribes a
  4-core laptop.

`profile()` reports what the machine has and what each stage should therefore be given;
`configure()` puts that into effect — environment variables for the libraries that read them,
direct calls for those that do not — and returns the profile it applied, so a run log records the
hardware a run *actually used* rather than the hardware it was asked for.

**Nothing here imports torch at module scope.** The API server, the report engine and the
verification suites all import `core.config` siblings on paths that must stay fast and must not
fail when the ML extras are absent.

`idle_gpu` is the detector for the silent case: an NVIDIA driver present, `torch.version.cuda`
empty, and the resolved device `cpu`. See §5 — `requirements.txt` pins the CPU wheel
deliberately, for the Space.

| Variable | Effect |
| :--- | :--- |
| `GROWNXT_DEVICE` | `auto` / `cuda` / `cuda:1` / `cpu` / `mps` / `xpu` |
| `GROWNXT_NUM_THREADS` | Threads for Docling and torch intra-op |

Guarded by `scripts/verify_core.py` (`check_hardware_resolution`, `check_idle_gpu_detector`).

> **Two knobs are now dead.** `GROWNXT_EMBED_BATCH_SIZE` and `GROWNXT_EMBED_FP16`, and the
> `embed_batch_size` / `embed_fp16` fields on `HardwareProfile`, sized a `SentenceTransformer`
> that no longer exists. F2 predicted this. Removing them changes a public dataclass, so it is a
> behaviour change and not part of the style refactor — tracked as follow-up.

> **`resolve_workers()` no longer exists.** F2 argued the GPU worker cap must survive the vector
> removal, but the function lived in `ingestion/batch.py` and went with it in `949240d`. The cap
> will have to be reintroduced wherever the qualitative pipeline gains concurrency; the reasoning
> stands, the code does not. CLAUDE.md's gotcha and this spec's F2 are corrected accordingly.
