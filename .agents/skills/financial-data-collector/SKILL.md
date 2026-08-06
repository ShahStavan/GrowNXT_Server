---
name: financial-data-collector
description: Discover stock listings, fetch 7-part financial statement bundles, compute institutional ratio analytics (5-Factor DuPont ROE, Solvency, Liquidity, ROIC, CAGR), download corporate filings, and serve ground-truth data via REST API.
---

# Financial Data Collector Skill

## Overview

The `financial-data-collector` skill provides automated workflows for discovering equity listings, ingesting fundamental financial statement datasets (Income Statements, Balance Sheets, Cash Flows, Stock Summary Profiles), computing institutional-grade financial ratios, downloading corporate filings, and exposing data over REST API endpoints.

---

## Service Endpoints & Capabilities

The Financial Data Collector service runs at:
`https://financial-data-collector-qrxj.vercel.app`

### 1. Granular Financial Statement Endpoints

| Endpoint Route | Method | Description |
| :--- | :---: | :--- |
| `GET /api/v1/health` | `GET` | Service health status and storage directory path. |
| `GET /api/v1/stocks/search` | `GET` | Searches stock ticker or company name (`?q={query}`). |
| `GET /api/v1/stocks/<symbol>/fetch` | `GET` | Downloads and saves complete 7-part statement bundle. |
| `GET /api/v1/stocks/<symbol>/financials` | `GET` | Serves aggregated JSON dataset for RAG ingestion. |
| `GET /api/v1/stocks/<symbol>/income/quarterly` | `GET` | Returns 8-quarter interim income statement (`quarterlyData`). |
| `GET /api/v1/stocks/<symbol>/income/annual` | `GET` | Returns 5-year annual income statement (`annualData`). |
| `GET /api/v1/stocks/<symbol>/income/quarterly/growth` | `GET` | Returns QoQ quarterly growth metrics with plain-English annotations. |
| `GET /api/v1/stocks/<symbol>/income/annual/growth` | `GET` | Returns YoY annual growth metrics with plain-English annotations. |
| `GET /api/v1/stocks/<symbol>/balancesheet` | `GET` | Returns annual balance sheet statements (`balancesheetData`). |
| `GET /api/v1/stocks/<symbol>/balancesheet/growth` | `GET` | Returns balance sheet solvency growth with plain-English annotations. |
| `GET /api/v1/stocks/<symbol>/cashflow` | `GET` | Returns annual cash flow statements (`cashflowData`). |
| `GET /api/v1/stocks/<symbol>/summary` | `GET` | Returns stock summary profile, business description, and market cap. |
| `GET /api/v1/stocks/<symbol>/peers` | `GET` | Returns list of peer companies with tickers, names, and SIDs. |

---

### 2. Institutional Ratio Analytics Endpoints

| Endpoint Route | Method | Derived Ratios Returned |
| :--- | :---: | :--- |
| `GET /api/v1/stocks/<symbol>/dupont` | `GET` | **Extended 5-Factor DuPont ROE Model**: Tax Burden, Interest Burden, Operating Margin %, Asset Turnover x, Equity Multiplier x. |
| `GET /api/v1/stocks/<symbol>/solvency` | `GET` | **Solvency & Coverage**: Interest Coverage Ratio (ICR), Net Debt, Net Debt-to-EBITDA x, Debt-to-Equity x. |
| `GET /api/v1/stocks/<symbol>/liquidity` | `GET` | **Working Capital Health**: Current Ratio x, Quick Ratio x, Receivable Days (DSO), Inventory Days (DIO). |
| `GET /api/v1/stocks/<symbol>/capital-efficiency` | `GET` | **Capital Allocation**: ROIC %, Free Cash Flow (FCF) Conversion %, Fixed Asset Turnover x. |
| `GET /api/v1/stocks/<symbol>/cagr` | `GET` | **Multi-Year Growth**: 3-Year & 5-Year Revenue CAGR %, EBIT CAGR %, PAT CAGR %. |

---

## Directory Data Schema

When a stock (e.g. `WIPRO`) is ingested, the files are stored as follows:

```
data/wipro/
├── sData.json          # Stock metadata, Sector, Market Cap, Peers list
├── quarterly.json      # Interim quarterly income statement (Sales, EBIT, PAT, EPS)
├── qtGrowth.json       # QoQ quarterly percentage growth metrics
├── annual.json         # 5-Year annual income statement
├── anGrowth.json       # YoY annual growth metrics
├── balancesheet.json   # Capital structure, Equity, Debt, Cash, Liabilities
├── balGrowth.json      # Balance sheet solvency growth trends
├── cashflow.json       # Operating cash flows, CapEx, and Free Cash Flow
├── annual_report.pdf   # Company Annual Report PDF (optional)
├── presentation.pdf    # Investor Pitch Presentation PDF (optional)
└── transcript.txt      # Earnings Conference Call Transcript (optional)
```
