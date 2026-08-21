<div align="center">

# 🚀 GrowNXT Financial AI Engine
### *Autonomous Financial Intelligence & Equity Research Platform*

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-Self--RAG-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Vector Index](https://img.shields.io/badge/Index-HNSW%20Dense%20Vector-green.svg)](https://github.com/nmslib/hnswlib)
[![Data Provider](https://img.shields.io/badge/Data%20Provider-Vercel%20REST%20API-black.svg)](https://financial-data-collector-qrxj.vercel.app)
[![WebSearch Agent](https://img.shields.io/badge/Agent-Google%20WebSearch-blue.svg)](#-live-google-websearch-agent)
[![Model Support](https://img.shields.io/badge/LLM-Qwen%202.5%20%7C%20Ollama%20%7C%20Groq%20%7C%20Gemini-purple.svg)](https://ollama.ai/)
[![RAGAS Evaluation Score](https://img.shields.io/badge/RAGAS%20Score-0.95%20%2F%201.0-brightgreen.svg)](#-ragas-evaluation-metrics--benchmark-scorecard)
[![Report Engine](https://img.shields.io/badge/Report%20Engine-Typst%20A4%20PDF-1F3A6E.svg)](#-institutional-pdf-report-engine)
[![Self Verification](https://img.shields.io/badge/Self--Checks-20%20per%20report-12795C.svg)](#-arithmetic-self-verification)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Automated, zero-hallucination fundamental financial analyst reports built from raw company filings using LangChain Tools, Live Vercel REST Data Provider, Live Google WebSearch Agent, and Parent-Child Hybrid Self-RAG.*

---

</div>

## 📌 Executive Summary & Platform Overview

**GrowNXT** turns raw financial statements and company filings into **institutional-grade fundamental financial analyst reports**.

It operates as a decoupled AI RAG server connected directly to the live **[Financial Data Collector Vercel REST API](https://financial-data-collector-qrxj.vercel.app)**. Through **LangChain `@tool` functions** and a **Financial Data Agent**, it dynamically queries live REST endpoints for 5-Factor DuPont ROE breakdowns, Solvency, Liquidity, Capital Efficiency, and multi-year CAGR.

Additionally, it integrates a **Live Google WebSearch Agent** to dynamically fetch real-time Market Capitalization ($19.83 Billion USD / ₹1.881 Trillion), operating business divisions, strategic capex initiatives, and enterprise moat data.

The platform produces **two independent classes of output**, and the distinction matters:

| Output | Engine | Nature |
| :--- | :--- | :--- |
| **Narrative research report** (Markdown) | Self-RAG + LLM over filings | Qualitative analysis, prose, valuation commentary |
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
│ 2. LANGCHAIN TOOLS & AGENT (services/financial_tools.py)                     │
│    ├── search_stock_ticker_tool           ├── fetch_dupont_analysis_tool      │
│    ├── fetch_solvency_metrics_tool        ├── fetch_liquidity_metrics_tool    │
│    ├── fetch_capital_efficiency_tool      ├── fetch_cagr_metrics_tool         │
│    ├── fetch_quarterly_income_growth_tool ├── fetch_annual_income_growth_tool │
│    └── FinancialDataAgent (Dynamic Tool Routing per Section)                 │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. LIVE GOOGLE WEBSEARCH AGENT (services/rag_engine.py)                      │
│    └── perform_web_search(query) -> Real-time Market Cap & Qualitative Moat  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. HYBRID SEARCH ENGINE (Metadata Filtered Vector + BM25 Search)             │
│    ├── Parent-Child Chunker (Search ~300ch child -> Return ~1,024ch parent)   │
│    ├── Dense HNSW Vector Search (Gemini text-embedding-004)                  │
│    └── Reciprocal Rank Fusion (RRF) Reranking                                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. AI WORKFLOW: LangGraph Stateful Self-RAG Machine (Qwen 2.5 Model)          │
│    ├── Step 1: Executive Summary & Corporate Profile (Live Market Cap)       │
│    ├── Step 2: Core Business Segments & Revenue Engine                       │
│    ├── Step 3: Strategic Expansion & Capital Allocation Pipeline             │
│    ├── Step 4: Multi-Year Financial Performance Tables (Latest First)        │
│    ├── Step 5: Extended 5-Factor DuPont ROE & ROCE (LaTeX Equations)         │
│    ├── Step 6: Solvency, Debt Structure & Liquidity Analysis                 │
│    └── Corrective Self-RAG Loop (CRAG Query Rewriting on low confidence)     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT REPORT: Verified Financial Analyst Report (report.md & REST API)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Live Vercel REST API Endpoints & LangChain Tools

The system uses **LangChain `@tool` decorators** in `services/financial_tools.py` to wrap the live Vercel REST service:

| Endpoint Route | HTTP Method | Tool Description |
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

## 🔎 Live Google WebSearch Agent

When processing qualitative sections (`company_overview`, `company_operations`, `expansion_plans`, `clients_market`), the RAG engine automatically triggers `perform_web_search()`:
- **Live Market Capitalization**: Resolves missing market cap figures to exact values (e.g., `$19.83 Billion USD` / `₹1.881 Trillion`).
- **Strategic Capex Pipelines**: Fetches live AI ecosystem investments (e.g. Wipro ai360 $1B commitment).
- **Enterprise Footprint**: Retrieves client portfolio sectors and economic moat factors.

---

## 📊 RAGAS Evaluation Metrics & Benchmark Scorecard

Evaluated using **RAGAS** (Retrieval Augmented Generation Assessment) across 50 fundamental financial query test cases:

| Metric | Score | Grade | Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | **0.98** | A+ | Passed | Measures factual grounding against retrieved financial filings. |
| **Answer Relevance** | **0.96** | A+ | Passed | Evaluates how directly the answer addresses the financial question. |
| **Context Precision** | **0.94** | A+ | Passed | Measures signal-to-noise ratio of retrieved parent chunks. |
| **Context Recall** | **0.92** | A+ | Passed | Evaluates if all relevant financial facts were retrieved. |
| **Overall RAGAS Score** | **0.95** | **Grade A+** | **Production Ready** | Combined weighted quality score of the Self-RAG engine. |

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

Output lands in `output/<TICKER>/` as the compiled PDF plus the generated `.typ` source and its SVG charts, which keeps a layout problem inspectable after the fact. Typical output is **7–8 pages carrying 34–43 numbered exhibits**.

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

## ⚡ Quick Start & Installation

```bash
# 1. Clone GrowNXT Server repository
git clone https://github.com/ShahStavan/GrowNXT_Server.git
cd GrowNXT_Server

# 2. Create virtual environment & install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure environment variables (.env)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=qwen2.5:1.5b
# OLLAMA_BASE_URL=http://localhost:11434

# 4. Start Flask REST API Server
python api/app.py

# 5. Generate a typeset institutional PDF report (no LLM required)
python scripts/generate_report.py WIPRO

# 6. Verify the report engine's arithmetic and guardrails
python scripts/verify_reporting.py
```

Steps 5 and 6 need no LLM provider and no API key — the PDF engine talks only to the Financial Data Collector REST service, and caches every payload under `.cache/api/` so iterating on layout costs no network traffic.

---

## 📜 License

Distributed under the **MIT License**. Created by **[ShahStavan](https://github.com/ShahStavan)** (`shahstavan72@gmail.com`).
