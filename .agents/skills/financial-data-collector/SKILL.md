---
name: financial-data-collector
description: Skill for discovering, fetching, validating, and serving fundamental financial stock data, filings, and institutional ratio analytics for any equity ticker.
---

# 📈 Financial Data Collector & Ingestion Skill

This skill provides step-by-step instructions, workflows, and API specifications for collecting, formatting, validating, and serving fundamental financial stock datasets (Income Statements, Balance Sheets, Cash Flows, Stock Summaries, and Annual PDFs) for any stock ticker.

---

## 🎯 Purpose & Scope

The **Financial Data Collector** is a decoupled standalone data service that powers downstream RAG pipelines (such as GrowNXT) by providing structured financial filings, unstructured PDF context, and derived institutional ratio analytics.

Key capabilities:
1. **Universal Stock Discovery**: Search any stock ticker/name and resolve unique Stock Identifiers (SIDs).
2. **Financial Dataset Fetching**: Retrieve 5-year Annual, 8-quarter Interim, Balance Sheet, Cash Flow, and YoY Growth metrics.
3. **Institutional Financial Analytics**: Calculate 5-Factor Extended DuPont ROE, Solvency Ratios, Working Capital Liquidity, ROIC & FCF Quality, and Multi-Year CAGR.
4. **Structured Storage**: Persist stock data into standardized JSON schemas (`sData.json`, `annual.json`, `quarterly.json`, `balancesheet.json`, `cashflow.json`).
5. **PDF Document Fetching**: Download company Annual Reports, Investor Presentations, and Conference Call Transcripts.
6. **Granular REST API Interface**: Expose REST endpoints to serve full or granular ground-truth JSON datasets directly to RAG engines.

---

## 🌐 Complete API Endpoints Specification

### 1. External Data Provider Endpoints (Tickertape & Screener)

| Category | Method | API Endpoint URL Template | Parameters / Description |
| :--- | :---: | :--- | :--- |
| **Search** | `GET` | `https://api.tickertape.in/search` | `text={query}&types=stock&pageNumber=0` |
| **Quarterly Financials** | `GET` | `https://api.tickertape.in/stocks/financials/income/{sid}/interim/normal` | `count=15` (8-quarter income statement) |
| **Annual Financials** | `GET` | `https://api.tickertape.in/stocks/financials/income/{sid}/annual/normal` | `count=15` (5-year annual income statement) |
| **Quarterly Growth** | `GET` | `https://api.tickertape.in/stocks/financials/income/{sid}/interim/growth` | `count=15` (QoQ growth metrics) |
| **Annual Growth** | `GET` | `https://api.tickertape.in/stocks/financials/income/{sid}/annual/growth` | `count=15` (YoY annual growth metrics) |
| **Balance Sheet** | `GET` | `https://api.tickertape.in/stocks/financials/balancesheet/{sid}/annual/normal` | `count=15` (Assets, Liabilities, Net Worth) |
| **Balance Growth** | `GET` | `https://api.tickertape.in/stocks/financials/balancesheet/{sid}/annual/growth` | `count=15` (Solvency & Capital Growth) |
| **Cash Flow** | `GET` | `https://api.tickertape.in/stocks/financials/cashflow/{sid}/annual/normal` | `count=15` (Operating, Investing, Financing CF) |
| **Stock Summary & Peers**| `GET` | `https://analyze.api.tickertape.in/v2/stocks/summary/{sid}` | Returns company overview, mcap & peer list |
| **Screener Search** | `GET` | `https://www.screener.in/api/company/search/` | `q={query}` (Secondary ticker lookup) |

---

### 2. Internal REST API Service Endpoints (Exposed for RAG Pipeline)

#### General & Aggregated Endpoints
- `GET /api/v1/health` — Service health check & storage status
- `GET /api/v1/stocks/search?q={query}` — Search stock symbol or company name
- `GET /api/v1/stocks/<symbol>/fetch` — Downloads & saves 7-part statement bundle for symbol
- `GET /api/v1/stocks/<symbol>/financials` — Serves aggregated JSON payload directly to the RAG

#### Granular Tickertape-Specific Endpoints
- `GET /api/v1/stocks/<symbol>/income/quarterly` — 8-Quarter Income Statement (`quarterlyData`)
- `GET /api/v1/stocks/<symbol>/income/annual` — 5-Year Annual Income Statement (`annualData`)
- `GET /api/v1/stocks/<symbol>/income/quarterly/growth` — QoQ Income Growth Metrics (Annotated with comments)
- `GET /api/v1/stocks/<symbol>/income/annual/growth` — YoY Annual Income Growth Metrics (Annotated with comments)
- `GET /api/v1/stocks/<symbol>/balancesheet` — Annual Balance Sheet Statements (`balancesheetData`)
- `GET /api/v1/stocks/<symbol>/balancesheet/growth` — Balance Sheet Solvency Growth (Annotated with comments)
- `GET /api/v1/stocks/<symbol>/cashflow` — Cash Flow Statements (`cashflowData`)
- `GET /api/v1/stocks/<symbol>/summary` — Tickertape Stock Summary Profile & Market Cap
- `GET /api/v1/stocks/<symbol>/peers` — List of Peer Companies (Tickers, Names, SIDs)

#### Institutional Derived Analytics Endpoints
- `GET /api/v1/stocks/<symbol>/dupont` — Extended 5-Factor DuPont ROE Model (Tax Burden, Interest Burden, Operating Margin, Asset Turnover, Leverage)
- `GET /api/v1/stocks/<symbol>/solvency` — Solvency Ratios (Interest Coverage Ratio, Net Debt, Net Debt/EBITDA, Debt-to-Equity)
- `GET /api/v1/stocks/<symbol>/liquidity` — Liquidity & Working Capital Health (Current Ratio, Quick Ratio, Receivable Days DSO, Inventory Days DIO)
- `GET /api/v1/stocks/<symbol>/capital-efficiency` — Capital Efficiency & Cash Quality (ROIC %, FCF Conversion %, Fixed Asset Turnover)
- `GET /api/v1/stocks/<symbol>/cagr` — Multi-Year Compound Annual Growth Rates (3-Year & 5-Year Revenue, EBIT, PAT CAGR)

---

## 📁 Standardized Stock Directory Schema

When data is collected for a stock (e.g., `ADANIENT`), it is organized under `data/adanient/`:

```
data/adanient/
├── sData.json          # Stock metadata, Sector, Industry, Market Cap, Peers
├── quarterly.json      # Interim quarterly income statement (Revenue, Operating Profit, PAT, EPS)
├── qtGrowth.json       # QoQ percentage growth metrics
├── annual.json         # 5-Year annual income statement
├── anGrowth.json       # YoY annual growth metrics
├── balancesheet.json   # Capital structure, Equity, Debt, Bank Cash, Liabilities
├── balGrowth.json      # Balance sheet growth trends
├── cashflow.json       # Operating cash flows & Capex allocations
└── annual_report.pdf   # (Optional) Company Annual Report PDF for vector indexing
```

---

## 🛠 Step-by-Step Workflow for Data Collection

### Step 1: Discover Stock Identifier (SID)
Use `FinancialSearchDiscovery` to resolve ticker symbol `WIPRO` to SID `WIPR`:
```python
from collector.search_discovery import FinancialSearchDiscovery
discovery = FinancialSearchDiscovery()
item = discovery.get_exact_match("WIPRO")
# Returns: {"sid": "WIPR", "name": "Wipro Limited", "sector": "Information Technology"}
```

### Step 2: Fetch and Save Complete Financial Bundle
Use `FinancialDataFetcher` to download all 7 JSON filings:
```python
from collector.fetcher import FinancialDataFetcher
fetcher = FinancialDataFetcher(output_dir="data")
success = fetcher.fetch_and_save_stock("WIPRO", sid="WIPR")
```

### Step 3: Compute Institutional Ratio Analytics
Use `FinancialMetricsEngine` to compute 5-factor DuPont ROE, Solvency, Liquidity, and ROIC:
```python
from collector.metrics_engine import FinancialMetricsEngine
dupont = FinancialMetricsEngine.compute_extended_dupont(annual_data, balance_data)
solvency = FinancialMetricsEngine.compute_solvency_metrics(annual_data, balance_data)
```

---

## ⚠️ Data Ingestion Rules & Standards
1. **Google Python Style**: Follow Google Python Style Guide with clear docstrings, type annotations, and safe type casting.
2. **Exact Currency Formatting**: Store raw numbers in JSON (e.g., `40950.56`). Formatting into `₹40,950.56 Cr` is handled by presentation layers.
3. **Reverse Chronological Sorting**: Ensure tables order latest periods first (FY 2026 -> FY 2025 -> FY 2024).
4. **Encoding Safety**: Save all files using `encoding='utf-8', ensure_ascii=False`.
