<div align="center">

# 🚀 GrowNXT Financial AI Engine
### *Autonomous Financial Intelligence & Equity Research Platform*

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-Ingestion%20Pipeline-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Data Provider](https://img.shields.io/badge/Data%20Provider-Vercel%20REST%20API-black.svg)](https://financial-data-collector-qrxj.vercel.app)
[![Model Support](https://img.shields.io/badge/LLM-Hosted%20Endpoint-purple.svg)](#-hosted-model-access)
[![Report Engine](https://img.shields.io/badge/Report%20Engine-Typst%20A4%20PDF-1F3A6E.svg)](#-institutional-pdf-report-engine)
[![Self Verification](https://img.shields.io/badge/Self--Checks-20%20per%20report-12795C.svg)](#-arithmetic-self-verification)
[![Ingestion](https://img.shields.io/badge/Ingestion-LangGraph%20%7C%20qwen3--embed-5B21B6.svg)](#-document-ingestion-layer-langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Automated, zero-hallucination fundamental financial analyst reports built from raw company filings using a Live Vercel REST Data Provider, a layout-aware document ingestion pipeline, and a deterministic typeset report engine.*

---

</div>

## 📌 Executive Summary & Platform Overview

**GrowNXT** turns raw financial statements and company filings into **institutional-grade fundamental financial analyst reports**.

It operates as a decoupled analysis server reading the live **[Financial Data Collector Vercel REST API](https://financial-data-collector-qrxj.vercel.app)** for 5-Factor DuPont ROE breakdowns, Solvency, Liquidity, Capital Efficiency, and multi-year CAGR, and the issuer's own filings for everything a statement cannot say.

The platform produces **two independent classes of output**, and the distinction matters:

| Output | Engine | Nature |
| :--- | :--- | :--- |
| **[Evidence dossier](#the-evidence-dossier)** (JSON + Markdown) | `ingestion/` — retrieval over the issuer's own filings | Cited passages from annual reports, calls and decks |
| **[Institutional PDF report](#-institutional-pdf-report-engine)** (A4, typeset) | `reporting/` — deterministic Python, **no LLM in the path** | Statements, ratios, composites, arithmetic verification |

The PDF engine never calls a language model. Every figure it prints is either reported by the data provider or computed in traceable Python, which is why it can carry an arithmetic self-verification appendix and a language model cannot.

---

## 🏛 System Architecture & Processing Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PROVIDER: Live Financial Data Collector Vercel REST Service          │
│    https://financial-data-collector-qrxj.vercel.app                           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. TYPED COLLECTOR CLIENT (reporting/client.py)                              │
│    ├── Fourteen statement, ratio and composite endpoints per symbol          │
│    ├── Parsed values with absence made explicit (a nil is not a zero)        │
│    └── On-disk payload cache (.cache/api) — a rebuild costs no requests      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. DOCUMENT INGESTION PIPELINE (ingestion/, LangGraph)                       │
│    catalogue -> download -> parse -> chunk -> embed -> prompt -> dossier     │
│    ├── Recursive XY-cut layout extraction (multi-column, A3 spreads)         │
│    ├── Type-aware parsing (annual report / transcript / investor deck)       │
│    └── qwen3-embed vectors + cited evidence dossier                          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. REPORT ENGINE (reporting/, deterministic — no LLM in the path)            │
│    ├── Ratios, composites (DuPont, Piotroski F, Altman Z), peer tables       │
│    ├── Native Typst markup + vector SVG charts                               │
│    └── Arithmetic self-verification appendix (20 checks per report)          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT: reports/<TICKER>_report.pdf — every stock's typeset A4 report     │
│            output/<TICKER>/ — charts, Typst source, caches, dossier          │
└──────────────────────────────────────────────────────────────────────────────┘
```

Generation itself goes through one hosted OpenAI-compatible endpoint
(`core/llm_config.py`); see [Hosted model access](#-hosted-model-access).

---

## 🌐 Collector REST Endpoints Consumed

`reporting/client.py` reads the following endpoints of the live Vercel service,
caching each payload under `.cache/api/<TICKER>/`:

| Endpoint Route | HTTP Method | Payload |
| :--- | :---: | :--- |
| `/api/v1/stocks/<symbol>/summary` | `GET` | Company profile summary & market peer list |
| `/api/v1/stocks/<symbol>/income/quarterly` | `GET` | 8-quarter interim income statements |
| `/api/v1/stocks/<symbol>/income/annual` | `GET` | 5-year annual income statements |
| `/api/v1/stocks/<symbol>/income/quarterly/growth` | `GET` | QoQ quarterly sales & PAT growth metrics |
| `/api/v1/stocks/<symbol>/income/annual/growth` | `GET` | YoY annual sales & PAT growth metrics |
| `/api/v1/stocks/<symbol>/balancesheet` | `GET` | Balance sheet assets & liabilities |
| `/api/v1/stocks/<symbol>/balancesheet/growth` | `GET` | Solvency & debt growth metrics |
| `/api/v1/stocks/<symbol>/cashflow` | `GET` | Operating & free cash flows |
| `/api/v1/stocks/<symbol>/dupont` | `GET` | Extended 5-Factor DuPont ROE Model ($\text{NPM} \times \text{Asset Turnover} \times \text{Leverage}$) |
| `/api/v1/stocks/<symbol>/solvency` | `GET` | Interest Coverage Ratio (ICR), Net Debt, D/E |
| `/api/v1/stocks/<symbol>/liquidity` | `GET` | Current Ratio, Quick Ratio, DSO, DIO |
| `/api/v1/stocks/<symbol>/capital-efficiency` | `GET` | ROIC %, FCF Conversion %, Fixed Asset Turnover |
| `/api/v1/stocks/<symbol>/cagr` | `GET` | 3-Year & 5-Year Revenue, EBIT, and PAT CAGR |

---

## 🤖 Hosted Model Access

Every generation call in the repository goes through one OpenAI-compatible
chat-completions endpoint (`core/llm_config.py`). The deployment selects the
model server-side, so nothing in this codebase names one, and there is no
provider to choose between — the earlier Ollama, Groq and Gemini branches are
gone along with the environment variables that selected them.

```python
from core.llm_config import LLMError, generate_llm_response

try:
    answer = generate_llm_response(prompt, context=evidence)
except LLMError as exc:      # request failed, or the completion came back empty
    ...
```

A failed call raises. The previous implementation returned the prompt's own
context when the endpoint was unreachable, which produced output that read like
analysis but was unprocessed source text — indistinguishable, to a caller, from
a real answer. `GROWNXT_LLM_API_KEY` is attached only when it is set, and
`GROWNXT_LLM_TIMEOUT` (default 120s) bounds the request.

---

## 📄 Institutional PDF Report Engine

`reporting/` renders a typeset A4 equity research document straight from the collector's REST payloads — no LLM anywhere in the path. Tables are emitted as native **Typst** markup and charts as **vector SVG**, so a figure stays sharp in print and every number on the page is traceable to a line of Python.

```bash
# Generate reports (reads .cache/api, add --refresh to re-request live data)
venv/Scripts/python.exe scripts/generate_report.py WIPRO
venv/Scripts/python.exe scripts/generate_report.py WIPRO RELIANCE TCS HDFCBANK

# Verify the engine's arithmetic and guardrails (exits non-zero on failure)
venv/Scripts/python.exe scripts/verify_reporting.py
```

The compiled PDF lands in `reports/<TICKER>_report.pdf`; the generated `.typ` source and its SVG charts stay in `output/<TICKER>/`, which keeps a layout problem inspectable after the fact. Typical output is **7–8 pages carrying 34–43 numbered exhibits**.

### Document structure

Sections are **semantic, not incidental** — each answers one question, and an exhibit only appears in the section whose question it addresses.

| # | Section | The question it answers |
| :---: | :--- | :--- |
| 1 | Earnings power | What the business earns, and the cost structure behind it |
| 2 | Near-term trajectory | Where momentum sits, with seasonality removed |
| 3 | Returns on capital | What it earns on capital tied up, and **which DuPont factor moved it** |
| 4 | Financial position | How the balance sheet is funded and how much room it leaves |
| 5 | Cash generation & earnings quality | Whether reported profit arrives as cash |
| 6 | **Capital allocation** | Where the cash went, and whether reinvestment earned its keep |
| 7 | **Composite quality & solvency scores** | What standard frameworks conclude, and on what evidence |
| 8 | Valuation | What the market pays, on equity and on the enterprise |
| 9 | Shareholder returns & ownership | What accrues per share, and who holds the register |
| 10 | **Verification** | The report's own arithmetic, re-derived and reported |

### Tier 1 — derived series

The collector's ratio endpoints each return a single trailing point. Tier 1 (`reporting/analytics.py`) turns those into series: margins, cost structure, ROE/ROCE/ROIC with NOPAT and invested capital, leverage and coverage, working-capital days, earnings quality and accruals, an EPS bridge decomposing growth into profit versus share count, and an enterprise-value bridge with EV multiples the provider does not publish.

### Tier 2 — composites

`reporting/composites.py` builds the composites that sit on top of those series. **Each is a published framework applied to the data; no coefficient is fitted here.**

| Composite | Components exposed | Presentation |
| :--- | :--- | :--- |
| **Piotroski F-Score** (2000) | All **9 signals**, each with its test, value, comparator and point | Table + score-history bars against the evaluated ceiling |
| **Altman Z-Score** (1968) | All **5 terms**, each with ratio × published coefficient = contribution | Table + Z-prime history shaded by distress / grey / safe band |
| **Sources & uses of cash** | 6 flow lines with amount and share, cumulative over the window | Paired tables + matched-length stacked bars |
| **Reinvestment identity** | NOPAT, capex, D&A, net capex, ΔWC, reinvested, rate, ROIC, implied vs delivered growth | Table + implied-against-delivered chart |
| **DuPont, every period** | 5 factors + ROE rebuilt + ROE direct + **residual** | Table + factors indexed to 100 at the window start |

Chart form is chosen per exhibit rather than by habit: bars for an integer signal count, a **banded line** for Altman because its thresholds are inherently spatial, **matched stacked bars** for sources-and-uses because they show the equality, and **indexed lines** for DuPont because its five factors sit at incompatible scales (0.69 against 22.49 against 1.60 for Wipro's latest year) and a shared raw axis would hide four of them.

### 🛡 Two engineering guardrails

These are enforced in code, asserted at runtime, and covered by the verification harness.

**1. A non-positive denominator returns `None`, never a number.**

Negative equity, negative EBITDA and a negative cost base all produce ratios that are arithmetically fine and analytically meaningless — and negative profit over negative equity yields a *positive* ROE that reads as strength on the page. `fmt.pos_div` / `fmt.pos_margin` guard the **denominator only**; a negative numerator over a positive base survives, because interest cover of −2.7× is the finding rather than an error. Applied to ROE, ROCE, ROIC, D/E, goodwill/equity, CFO/PAT, capex/D&A, the effective tax rate (a pre-tax loss would otherwise corrupt NOPAT and ROIC downstream), net debt/EBITDA, EV multiples and the working-capital day counts.

**2. A composite is never reported as a bare number.**

"F-Score 7" is not analysis; the nine sub-tests are. Every composite carries its components as data, and the total is a derived *property* of the structure holding them — so the renderer cannot print a score without the parts in hand. Scores print as **"5 of 9"**, never a bare numerator, and the point column visibly sums to the total printed beneath it.

Three departures from the textbook are declared on the page rather than buried:

- **F-Score signal 8** substitutes EBITDA margin for the paper's gross margin, because this provider's gross-profit field is null for every ticker and its raw-material line covers materials only (64% of revenue for a refiner, under 1% for a services business).
- **F-Score** scales profitability by *opening* total assets as the 2000 paper specifies, which deliberately differs from the period-end convention the returns section uses to match the provider's own endpoints.
- **Altman and the reinvestment identity are withheld entirely for financials**, where working capital and sales-over-assets do not describe a lender. Two of the nine F-Score signals are withheld for the same reason and the score is then reported out of 7, flagged as not comparable with a standard F-Score.

The engine also names an Altman score dominated by one term: TCS scores 10.999 with **6.863 of it from market capitalisation over liabilities**, which is a valuation observation rather than a solvency one. The exhibit says so and tells the reader to read the term, not the total.

---

## ✅ Arithmetic Self-Verification

`reporting/selfcheck.py` re-derives what can be re-derived by an independent route and prints the residuals as a **Verification** section in the report itself. A report that shows its residuals is making a checkable claim; one that stays silent is asking for trust.

| Check type | Meaning | Tolerance |
| :--- | :--- | :--- |
| **Identity** | A definition, so it must close to floating-point precision | `1e-6` pp / `0.5` Rs cr |
| **Reconciliation** | A figure computed here against the provider's published value | One unit in the provider's last published decimal place |
| **Guardrail** | An editorial rule asserted to have held in practice | Zero breaches |

Identities verified include the cash flow statement articulating to its own reported movement in cash, free cash flow equalling operating cash less capex, **sources equalling uses**, the five DuPont factors multiplying to ROE, each composite total equalling the sum of its components, and implied growth equalling reinvestment rate × ROIC.

**Results across the four verified tickers:**

| Ticker | Sector | Checks | Result | Withheld |
| :--- | :--- | :---: | :---: | :--- |
| WIPRO | IT Services | 20 | **20 / 20** | — |
| RELIANCE | Oil & Gas Refining | 20 | **20 / 20** | — |
| TCS | IT Services | 20 | **20 / 20** | — |
| HDFCBANK | Private Banks | 15 | **15 / 15** | Z-Score, reinvestment identity, F-Score signals 5–6 |

Largest identity residual observed across all four: **5.8e-11**, on HDFCBANK; the three non-financials close at **1.8e-15 to 1.4e-14**. Worst reconciliation consumed **49%** of its rounding budget, so the tolerances are not masking a methodology difference.

### Verification harness

`scripts/verify_reporting.py` runs **7 checks** and gates a commit. The important one exists because none of the cached companies has negative equity — every real report takes the happy path, so a passing report proves nothing about what happens when a balance sheet turns. The harness therefore builds a **synthetic distressed company** (negative equity, EBITDA, pre-tax profit and cost base simultaneously) and asserts 11 ratios come back absent, that the F-Score still scores (2 of 9) and Altman lands in the distress band (−1.05), and that the full report still compiles when most figures are missing.

It also asserts the **breach detector reports a planted breach**, because a check that cannot fail is not evidence.

---

## 📥 Document Ingestion Layer (LangGraph)

`ingestion/` turns a company's published filings into embedded, citable evidence. It is the qualitative counterpart to the PDF engine: where `reporting/` computes from the collector's structured payloads, this layer reads what the company actually *said* — in its annual report, on its earnings calls, and in its investor decks — and prepares it for retrieval.

```
 catalogue ─▶ download ─▶ parse ─▶ chunk ─▶ embed ─▶ prompt ─▶ dossier
     │            │           │        │        │         │          │
  Screener     SHA-256    layout-   section- qwen3-    per-class  pivoted
  document     hashed,    aware     aware,   embed     template,  into the
  catalogue    idempotent recursive citable  4096-dim, cited      report's
                          XY-cut    chunks   float32   evidence   sections
```

```bash
# Ingest the latest annual report, transcripts and decks for one ticker
venv/Scripts/python.exe -m scripts.ingest_documents WIPRO

# Three years of annual reports, transcripts only
venv/Scripts/python.exe -m scripts.ingest_documents WIPRO --annual-reports 3 --doc-types concall_transcript

# Skip the compliance bulk of an annual report (notice, governance, ESG)
venv/Scripts/python.exe -m scripts.ingest_documents WIPRO --skip-sections notice governance esg

# Inspect what is already ingested, without touching the network
venv/Scripts/python.exe -m scripts.ingest_documents WIPRO --status

# Verify the layer, including the run-twice guarantee (exits non-zero on failure)
venv/Scripts/python.exe -m scripts.verify_ingestion WIPRO
venv/Scripts/python.exe -m scripts.verify_ingestion --offline   # no network, no API
```

Everything lands under `output/<TICKER>/` — the same per-stock workspace the PDF engine builds in: the filings in `documents/`, the parse and chunk caches in `parsed/` and `chunks/`, the vectors in `vectors/`, the built prompts and the [evidence dossier](#the-evidence-dossier) in `findings/`, and the state that ties them together in `registry.json`.

### The registry, and why a second run does nothing

Re-ingesting a company is expensive — one annual report is a 9 MB download, three minutes of PDF parsing, and 2,579 chunks to embed. So every stage records the **identity of the input it consumed**, and a stage runs only when that identity changes:

| Stage | Re-runs when | Because |
| :--- | :--- | :--- |
| `download` | the catalogue's URL for the document changes | Screener publishes a new link when the issuer files a new document — the link is the only signal available |
| `parse` | the downloaded bytes change, or that document class's parser version changes | parser versions are **per class**, so fixing the transcript parser does not re-embed the annual report |
| `chunk` | the parse output or the chunker's parameters change | |
| `embed` | the chunk set changes, or the embedding model changes | vectors from two models are not comparable, and mixing them returns confident nonsense with no trace in the output |
| `prompt` | the vectors or the company name change | |

A stage is also re-run when an output file it claims to have written has gone missing. The registry is written atomically, so an interrupted run cannot strand documents already on disk, and a corrupt registry is preserved rather than overwritten — losing it would re-download and re-embed everything.

```
$ python -m scripts.ingest_documents WIPRO      # first run: 8 filings, 494 vectors
[WIPRO] run complete in 137.0s: {"download": {"current": 8}, "parse": {"parsed": 8},
                                 "chunk": {"chunked": 8}, "embed": {"embedded": 8,
                                 "vectors": 494}, "prompts": {"built": 8},
                                 "dossier": {"written": 1}}

$ python -m scripts.ingest_documents WIPRO      # second run
[WIPRO] plan: nothing to do; every document is current.
[WIPRO] run complete in 1.8s: {"download": {"current": 8}, "parse": {"current": 8},
                               "chunk": {"current": 8}, "embed": {"current": 8},
                               "prompts": {"current": 8},
                               "dossier": {"unchanged": 1}}

$ python -m scripts.ingest_documents WIPRO --force chunk   # re-chunk, don't re-embed
[annual_report_FY2026] chunks are unchanged; keeping existing vectors.
[WIPRO] run complete in 2.4s: {"chunk": {"chunked": 8}, "embed": {"current": 8}, ...}
```

### Layout-aware extraction

Annual reports are typeset in two and three columns, and Indian issuers publish A3 spreads holding two logical pages side by side. A naive `extract_text()` walks such a page across its full width, splicing the left column into the right column's mid-sentence:

> 26) As the 80th AGM is being held through VC, the route **experience in the industry, his involvement in the** map is not annexed to this Notice. **operations of the Company over a long**

Chunks built from that are unusable as evidence. `ingestion/layout.py` runs a **recursive XY-cut** instead: it finds the whitespace gutters of a block, assigns each line to a column, emits the columns in reading order, and recurses into each one — because at the top level of a spread the only gutter wide enough to survive is the one *between the two pages*, and every threshold has to be re-measured against one page's width before that page's own columns become visible.

Three filing-specific behaviours fall out of the same pass:

- **Gutters are found by ink tolerance, not by a strict scan or a majority vote.** A strict "no row prints here" scan is defeated by the one centred heading that crosses the gutter; a per-row majority vote is defeated from the other side, because columns set with different leading share no baselines, so *no* row crosses the gutter for a majority to be drawn from. Counting rows with ink at each position and allowing a minority of them survives both.
- **Table rows are never cut at a gutter.** A statement of cash flows printed beside two prose columns puts its labels on one side of the gutter and its figures on the other; splitting there files the figures away from the line item they belong to.
- **A full-width element cuts only the columns it covers.** On a spread, the left page's tables must not interrupt the right page's prose, which is a separate block of reading.

Rotated sidebar furniture (`TROPER LAUNNA DETARGETNI ORPIW`) and folio numbers are dropped — the latter because a page number parked in the outer margin stretches the text block by 40 points, and the empty band beside it then reads as a gutter.

### Nothing is tied to one issuer

The layer ingests any listed Indian company; no module names a specific filer, in code or in comment, and a verification check scans the package against a list of 35 large-cap issuers to keep it that way. The behaviours that would otherwise invite a hardcoded name are derived instead:

| What varies by issuer | How it is handled |
| :--- | :--- |
| The overview heading is the company's own name — *About <name>*, *<name> at a glance* | `company_keywords()` builds those phrases from the catalogued company name at parse time. An unknown name yields no keywords rather than matching *"About Limited"* |
| A covering letter signs off *For <Issuer> Limited*, which looks like a speaker label | Matched by shape — `For` plus a short phrase ending in a corporate suffix — never against a list of names |
| Filings arrive from an exchange, from the issuer's own CDN, and from registrars | Download headers derive their referer from the URL's own host, rather than pinning one exchange's |
| The rupee is written `₹`, `Rs.` or `INR`, sometimes within one document, and a mark may be its own token | All three forms are recognised as table cells, and a standalone currency mark is transparent to the row scan |
| Symbols carry punctuation | `safe_ticker()` folds anything that is not alphanumeric, `-`, or `_`; the symbol is percent-encoded in every request, so an `&` cannot terminate a URL path |
| Section names differ across issuers | The vocabulary carries the common Indian variants — *Statutory Reports*, *Chairman's Statement*, *Notes forming part*, *Risks and Concerns*, *Value Added Statement*, the six capitals of an integrated report — and every page inherits the previous page's section when it carries no signal of its own |

Two of these were real defects found by this audit, not hypotheticals: an `about <issuer>` phrase in the section vocabulary, and a `for <issuer> limited` entry in the transcript's non-speaker labels. Both worked for one company and would have silently mis-parsed every other.

**Section keywords may not overlap.** Sections are chosen by keyword score, so a phrase listed under two of them makes the winner arbitrary. Listing *"management discussion and analysis report"* under governance as well as MD&A is defensible — the MD&A is formally an annexure to the Board's Report in an Indian filing — and it put pages of one issuer's MD&A into `governance`, which the default skip list drops. Two checks now assert that no phrase is claimed by two sections and that no phrase contains another section's phrase.

**Headings are looked for below the top of the page, and only the first two count.** The XY-cut emits the running head and any sidebar column before the body, so a section's own title routinely lands well down the reading order: one mid-cap prints `MANAGEMENT DISCUSSION AND ANALYSIS REPORT:` as the ninth line of its page. Scanning only the first four lines missed it, and that whole section inherited the label of the section before it.

Scanning the first fourteen lines finds it. Letting *every* heading on a page announce a section does not work either — a "Climate Action" panel inside a business narrative moved page runs into `esg` — so only a page's first two headings may announce, and a page announcing three or more sections is treated as a table of contents rather than the start of any of them.

Measured on three reports, that combination found a section that had been invisible and stopped carry-forward from mislabelling statutory pages:

| | `mdna` chars | Narrative retained | What changed |
| :--- | ---: | ---: | :--- |
| Mid-cap FMCG, 382 pp | 0 → **41,275** | 21.0% → 8.3% | MD&A found; statutory pages that carry-forward had been sweeping into `business` now correctly `governance` |
| Large-cap IT, 561 pp | 55,229 (unchanged) | 16.7% → 18.2% | prints proper running heads, so never depended on the deeper scan |
| Large-cap bank, 678 pp | 51,112 → **58,095** | 19.2% → 21.8% | `risk` recovered from 31k to 83k |

The narrative figure *falling* for the first of those is the point: it is misclassified statutory content leaving the index, not narrative being lost. Labels inside a statutory block remain approximate — a few page runs there land in `notice` or `financials` rather than `governance` — but all three are skipped by default, so it changes nothing that gets embedded.

```bash
# Any Indian symbol; the same pipeline, the same guarantees
venv/Scripts/python.exe -m scripts.ingest_documents BIKAJI
venv/Scripts/python.exe -m scripts.ingest_documents HDFCBANK
venv/Scripts/python.exe -m scripts.ingest_documents M&M
```

### Type-aware parsing

Flattening three document classes into one bag of pages throws away exactly the metadata that later makes retrieval precise.

| Class | Parsed into | Why it matters |
| :--- | :--- | :--- |
| **Annual report** | pages classified into `mdna`, `business`, `risk`, `financials`, `governance`, `esg`, `notice`; tables kept whole | an annual report is a bound volume of unrelated documents. Answering "revenue growth commentary" from the AGM notice is a wrong answer that reads like a right one. Issuers print the section name as a running head on every page, which beats any font-size heuristic |
| **Concall transcript** | `prepared_remarks` and `qa`, split at the handover, one block per speaker turn with a name and a role | management's framing and an analyst's scepticism carry different evidentiary weight, and a report must not quote one as the other. The handover is spoken by whoever holds the call — Wipro's CFO ends with *"with that, I will hand it over for Q&A"* — so both the moderator's opening and management's hand-off are matched, and how the boundary was found is recorded rather than assumed |
| **Investor presentation** | one block group per slide, each carrying its slide title | a slide title is often the only thing that says what its figures measure |

Chunks are packed to ~1,400 characters but never cross a section, never mix two speakers, and never separate a table from its rows. Fragments with no content are dropped rather than stored. Each carries a **context header** into its embedding text — ticker, document class, period, section, speaker — because an isolated paragraph reading *"margin declined 120 basis points sequentially"* is nearly unretrievable when nothing in it says which company, which period, or who said so.

### Embeddings

Chunks are embedded with **`qwen3-embed`** (4096 dimensions) through the hosted OpenAI-compatible endpoint, L2-normalised so cosine similarity is a dot product, and stored as raw `float32` beside a manifest naming the model, the dimension, and the chunk id of every row. As JSON, one annual report's vectors would be hundreds of megabytes and seconds of parsing per query; as binary they are 16 KB per vector and a single `np.fromfile`.

Retrieval refuses to load a vector file whose manifest disagrees with the chunk cache. The rows are positional, so continuing would pair every chunk with another chunk's vector and return fluent nonsense at ordinary-looking similarity scores.

### What is embedded, and what is not

An integrated annual report is mostly not an equity research document. Wipro's FY2026 filing chunks into 2,579 passages, and **1,954 of them are the audited statements and notes to accounts** — figures the data provider already publishes and the deterministic engine already typesets with an arithmetic audit trail. Embedding them costs most of the ingest's wall clock and makes retrieval *worse*, because they compete for top-k against the ~230 chunks of narrative a thesis is actually written from.

So four sections are skipped by default — `financials`, `notice`, `governance`, `esg` — and the narrative sections never are:

| | Chunks embedded | Wall clock | Vectors on disk |
| :--- | ---: | ---: | ---: |
| Whole volume | 2,940 | 1,115s | 47 MB |
| Default (compliance skipped) | 494 | 137s | 9.5 MB |

Pass `--skip-sections` with no values to embed everything, or name your own set. `--skip-sections notice governance` keeps the ESG report for an ESG mandate.

Embedding batches are issued from a small thread pool (`GROWNXT_EMBED_CONCURRENCY`, default 4). Measured against the live endpoint on four batches of 64 chunks: **33.7s sequential, 12.3s at four workers, 12.2s at eight** — the endpoint saturates at four, so raising it further buys nothing. Output is byte-identical either way, so changing concurrency never invalidates a stored vector.

A stage also declines to run when its input has not really changed, not merely when the stage above it re-ran. Changing the section default re-chunks all eight filings but leaves the transcripts' chunks identical, so only the annual report is re-embedded:

```
$ python -m scripts.ingest_documents WIPRO --force chunk
[transcript_2026_07] chunks are unchanged; keeping existing vectors.
```

### When a source or the service refuses

Ingesting across many issuers means meeting every way a filing can be unavailable, and each failure is contained to one document -- the run continues, the registry records the reason, and the next run retries only what failed.

| Failure | What happens |
| :--- | :--- |
| An investor-relations link now returns an HTML error page with a `200` | Refused: a response is accepted only if it begins with a PDF header, so an error page can never be filed as a filing and parsed into nonsense |
| A CDN blocks automated requests with `403` | Recorded against that document; the other filings complete. Verified not to be a referer problem -- the host refuses with a same-origin referer, an exchange referer, and none at all |
| The embedding endpoint's content filter refuses a batch | The batch is **bisected** to isolate the passage, which is excluded and named in the manifest; the rest of the document is embedded |

That last one was worth the code. The filter applies to the whole request, so one passage it dislikes fails the batch of 64 around it -- and a bank's investor deck naming a security incident refused three documents outright, discarding 129 usable passages between them. A permanent rejection (`4xx` other than `408`/`429`) is now a property of the *input*, not the moment: retrying is pointless, so the batch splits until the cause is isolated, at a cost of about `log2(batch)` extra requests.

Excluded chunks are named rather than dropped silently. A passage missing from the index is unretrievable, and a reader comparing the chunk cache against the manifest needs to see that the gap was deliberate:

```json
{ "count": 42, "excluded_chunk_ids": ["presentation_2026_07::slides::018"],
  "exclusion_reason": "HTTP 400: Content blocked: harmful_violence ..." }
```

The live verification asserts that the manifest accounts for every chunk -- embedded, or named as refused -- because vector rows are positional and an unexplained gap would shift every row after it.

### The evidence dossier

The pipeline's last stage pivots the per-document evidence into the report's own structure. `ingestion/dossier.py` writes two files to `output/<TICKER>/findings/`:

- **`evidence_dossier.json`** — what the report generator consumes, keyed by the eight report section keys.
- **`evidence_dossier.md`** — the same content as a readable, citable document, so a person can check that the evidence behind a section is the evidence they would have chosen.

It is *derived*, not re-retrieved: everything comes from the prompt artifacts and chunk caches already on disk, so building it costs no embedding calls and is reproducible from a completed ingest.

```
$ python -m scripts.ingest_documents WIPRO
[WIPRO] dossier: 95 passages across 8 of 8 report sections
```

Two rules make it a research input rather than a text dump. **No document may crowd out the others** — an annual report has an order of magnitude more chunks than an earnings call, so on score alone it wins every section and buries the more recent, more specific commentary; a cap per document per section keeps the latest call present. And **gaps are stated**: a section with no evidence is recorded as empty, and a target that retrieved nothing is named, because "the filings are silent on this" is a finding and is not the same as "the ingest skipped it".

Every passage carries the citation it was retrieved with:

```markdown
## 5. Financial performance and growth

**1. Earnings Call Transcript · Oct 2025 · Question and Answer Session · p8 · Aparna Iyer (management)**
_margin_levers · similarity 0.621_

> ...operationally, too, our utilization has improved and attrition has come
> down. We also drove better profitability in our fixed price program... our
> endeavor will be to be in a narrow band of our adjusted operating margins
> of 17.2. The notable one-off was the provision for bad and doubtful debt...

`transcript_2025_10::qa::014`
```

### A retrieval bug worth recording

The first dossier built put this at the top of *financial performance*, at 0.922 similarity:

> 2 2

Two compounding causes, both mine. Each chunk's embedding text opens with a context header — ticker, document class, period, section, speaker — which is what makes an isolated paragraph retrievable at all. But the probe queries **mirrored that same header**, so similarity was scoring header against header for every candidate, and the chunks that win such a comparison are the ones with the least content: a near-empty chunk *is* its header. Compounding it, chunks with no content were being stored at all — stranded chart data labels, and call turns reading `Thank you so much and all the very best`.

Both are fixed, and the fix order matters. Queries now carry the topic only (`ingestion/prompts.py`); the document is already narrowed to one filing and one section set before any probe runs, so the company and period added nothing the search did not already know. And `has_substance()` drops fragments at chunk time — 35 from the annual report, 12 to 18 per transcript — since a retrieval-time filter would still have to rank them first in order to exclude them.

Similarity scores fell from ~0.92 to ~0.62 as a result, which is the healthier number: they now measure content rather than a shared prefix.

### Extraction prompts

One generic template serves all three classes. What differs is not the task — extract what a research analyst would carry forward — but what each source can be trusted to establish, so the class-specific part is a **profile**: what the document is authoritative for, what it is not, which topics to look for, and how a claim from it must be attributed.

```
=== WHAT THIS SOURCE ESTABLISHES ===
An audited, board-approved filing. Figures in the financial statements and the
segment tables are the company's official record for the financial year...

=== WHAT IT DOES NOT ESTABLISH ===
It is retrospective and written to be favourable. The narrative sections select
which facts to emphasise, forward-looking statements are aspirations rather
than guidance...
```

Everything else is assembled from the document actually in hand. A static prompt asks an annual report about its risk section whether or not one was found, invites the model to fill the gap, and gets a fabrication that looks exactly like a real finding — so the targets are filtered to the sections the parser produced, the retrieval probes carry the company's own name and period (matching the chunks' context headers, which is what makes them retrieve precisely), and the numbered evidence block forces every finding to cite the chunk it came from.

Findings are returned as JSON routed to the report generator's own section keys (`company_overview`, `financial_results`, `dupont_analysis`, …), under rules that name the failure modes directly: reproduce figures exactly as printed, **do not compute derived metrics** — ratios belong to the deterministic engine, where they can be verified — attribute management's claims as claims, and **name what is missing**, because an honest absence is more useful than a thin inference and far more useful than a plausible invention.

### Verification

`scripts/verify_ingestion.py` checks the phase's acceptance criterion directly — the pipeline is run twice against a temporary data directory and the second run's own action counters are asserted, rather than inferred from log output. **107 checks**, 85 of which need no network.

The rest cover the invariants that make the recorded state trustworthy — each one a way the layer could appear to work while being wrong:

- vectors and chunks agree positionally, and concurrent batches reassemble in submission order (a reordering here pairs every chunk with another chunk's vector while leaving every count and checksum plausible)
- every embedding names the model that produced it
- a changed catalogue URL invalidates the stages below it
- a probe for an absent section returns nothing, rather than the nearest available text
- re-chunking does not re-embed when the chunks come out identical
- no stored chunk is a content-free fragment, and no probe repeats the chunks' context header
- a transient batch failure aborts its document rather than writing a vector set with an unexplained hole in it, while a *refused* passage costs one passage and is named in the manifest
- no ingestion module names a specific issuer, in code or in comment — scanned against 35 large-cap symbols
- no section keyword is claimed by two sections, and none contains another section's keyword

```bash
venv/Scripts/python.exe -m scripts.verify_ingestion BIKAJI     # 107 checks
venv/Scripts/python.exe -m scripts.verify_ingestion --offline  # 85, no network
```

---

### Where artifacts land

| Path | Holds |
| :--- | :--- |
| `reports/<TICKER>_report.pdf` | Every stock's finished report, together in one directory |
| `output/<TICKER>/` | That stock's build workspace: chart SVGs, the Typst source, the ingestion caches and evidence dossier, and the Drive upload record |
| `.cache/api/<TICKER>/` | Cached collector payloads, so a rebuild costs no requests |

Both roots are configurable (`GROWNXT_REPORTS_DIR`, `GROWNXT_OUTPUT_DIR`). The
split exists because Typst resolves `#image` paths relative to its source file
and chart names repeat across tickers -- so the workspace stays per-stock, while
the artifact a person actually wants is collected in one place.

---

## ☁️ Google Drive Delivery

`storage/gdrive.py` mirrors each compiled report into Google Drive, shared as
readable by anyone with the link, and `GET /api/stocks/<symbol>/report` returns
that link. Free-account storage is 15 GB shared with Gmail and Photos, which is
roughly 14,000 reports at ~700 KB each.

```bash
# One-time authorisation (opens Google's consent screen, stores the grant)
venv/Scripts/python.exe -m scripts.gdrive_auth

# Confirm the stored grant still works, without re-consenting
venv/Scripts/python.exe -m scripts.gdrive_auth --check

# Verify the whole delivery path offline -- 32 checks, no credentials needed
venv/Scripts/python.exe -m scripts.verify_gdrive
```

Setup, once, in the Google Cloud console: create a project, enable the **Google
Drive API**, create an **OAuth client ID of type Desktop**, and put the id and
secret in `.env` as `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET`. Optionally set
`GDRIVE_FOLDER_ID` to a folder; without it, reports land in the account root.
`scripts/gdrive_auth.py` then writes the refresh token to `.gdrive_token.json`,
which is git-ignored because it is a live credential.

### Publish the consent screen, or it breaks after a week

Access tokens last an hour and are refreshed automatically, so a running server
needs no attention. A **refresh** token is different: while the OAuth consent
screen sits in *Testing* status, Google expires it after **7 days**, and minting
a new one requires a human at the consent screen -- no code can do that
unattended. **Publishing the consent screen to Production removes that clock**,
which turns `gdrive_auth` into a once-ever step.

Everything either side of that is handled:

| Situation | Behaviour |
| :--- | :--- |
| Access token expired (hourly) | Refreshed silently before the next call, with a two-minute margin so a slow upload cannot race the expiry |
| Drive answers `401` mid-request | Token re-minted and the call retried once |
| Google rotates the refresh token | The new one is written to the token file, so a restart still works |
| Grant revoked or expired | `DriveAuthError`, and the API answers **503** naming the re-auth command rather than quietly returning no link |
| Drive down, or a network fault | `DriveError`, and the API answers **502**; the PDF endpoint still serves the report |

### Why a service account is not used

A service account has no Drive storage of its own and cannot own files -- an
upload to My Drive fails with `403 storageQuotaExceeded` even on an empty
account. Owning files requires either a Shared Drive or domain-wide delegation,
both of which need Google Workspace. An OAuth grant against a real account is
what works on a free plan.

### What the upload does, and does not, repeat

A report is uploaded once. The PDF's SHA-256 is recorded in
`output/<TICKER>/drive.json`, and a request whose PDF hashes the same is
answered from that record with no Drive traffic at all. A rebuilt report
**replaces the Drive file in place** rather than adding a second copy, so a link
already shared with someone keeps resolving to the current report -- Drive
permits duplicate names, so an upload that did not look for its predecessor
would accumulate one copy per rebuild.

One caveat worth knowing before this becomes the primary viewer: Drive is not a
CDN. Public links hit hard enough trip Google's anti-abuse lock -- the
"download quota exceeded" error, triggered by request frequency -- and there is
no dial to raise. The `/report/file` endpoint remains as the path that no
external quota can throttle.

---

## ⚡ Quick Start & Installation

```bash
# 1. Clone GrowNXT Server repository
git clone https://github.com/ShahStavan/GrowNXT_Server.git
cd GrowNXT_Server

# 2. Create virtual environment & install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure environment variables (.env) — all optional
# GROWNXT_LLM_API_URL=https://grownxt-llm.vercel.app   # hosted model endpoint
# GROWNXT_LLM_API_KEY=...                              # only if the endpoint requires one
# GROWNXT_OUTPUT_DIR=/var/lib/grownxt/output           # per-stock build workspace
# GROWNXT_REPORTS_DIR=/var/lib/grownxt/reports         # finished PDFs, all stocks
# GDRIVE_CLIENT_ID=...                                 # Drive delivery (see above)
# GDRIVE_CLIENT_SECRET=...
# GDRIVE_FOLDER_ID=...                                 # optional destination folder

# 4. Start the Flask API server (search + PDF delivery)
python api/app.py

# 5. Generate a typeset institutional PDF report (no LLM required)
python scripts/generate_report.py WIPRO

# 6. Verify the report engine's arithmetic and guardrails
python scripts/verify_reporting.py

# 7. Ingest a company's filings (annual report, transcripts, decks)
python -m scripts.ingest_documents WIPRO

# 8. Verify the ingestion layer, including the run-twice guarantee
python -m scripts.verify_ingestion WIPRO

# 9. Authorise Google Drive delivery, then verify it offline
python -m scripts.gdrive_auth
python -m scripts.verify_gdrive
```

### Server routes

| Route | Returns |
| :--- | :--- |
| `GET /api/search?q=<query>` | Matching listed companies (min. 3 characters) |
| `GET /api/stocks/<symbol>/report` | JSON carrying the **Google Drive link** to the report, compiled and uploaded on first request. `?refresh=1` rebuilds it and replaces the Drive file in place |
| `GET /api/stocks/<symbol>/report/file` | The PDF bytes themselves, for clients not using Drive. `?download=1` sends it as an attachment instead of inline |

```bash
curl "http://localhost:5000/api/search?q=wipro"
curl "http://localhost:5000/api/stocks/WIPRO/report"
curl -o WIPRO.pdf "http://localhost:5000/api/stocks/WIPRO/report/file"
```

```json
{
  "success": true, "symbol": "WIPRO", "generated": false,
  "view_link": "https://drive.google.com/file/d/<id>/view",
  "preview_link": "https://drive.google.com/file/d/<id>/preview",
  "drive": { "file_id": "<id>", "shared": true, "uploaded_at": "..." },
  "pdf_endpoint": "/api/stocks/WIPRO/report/file"
}
```

`preview_link` is the one to drop into an `iframe`; `view_link` is Drive's own
viewer page.

Steps 5 and 6 need no LLM provider and no API key — the PDF engine talks only to the Financial Data Collector REST service, and caches every payload under `.cache/api/` so iterating on layout costs no network traffic.

---

## 📜 License

Distributed under the **MIT License**. Created by **[ShahStavan](https://github.com/ShahStavan)** (`shahstavan72@gmail.com`).
