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
| `GET /api/v1/stocks/<symbol>/documents` | `GET` | Returns the corporate filing **link catalog** — annual reports, concall transcripts, investor presentations. Windowed by `?annual_reports={n}` (default 1) and `?concall_years={n}` (default 3). |

---

### 2. Document Link Catalog

`GET /api/v1/stocks/<symbol>/documents`

Division of responsibility: **Tickertape** supplies structured statement JSON,
**Screener.in** supplies document links. The service returns links only — it
never downloads files. The GrowNXT ingestion layer decides what to fetch,
hash, parse, and embed.

The two document groups are windowed **independently**, because they are
consumed differently. An annual report is one authoritative document per
financial year and the ingestion layer almost always wants the newest, so the
default is the latest one alone. Transcripts and presentations are read as a
series, so the default there is three years.

| Query Parameter | Default | Description |
| :--- | :---: | :--- |
| `annual_reports` | `1` | Annual reports to return, newest first. `0` returns every report available (capped at 25). |
| `concall_years` | `3` | Lookback window in years for concall transcripts and presentations. `0` returns every concall available (capped at 15 years). |
| `years` | — | Legacy alias applying one window to both groups. Either parameter above takes precedence for its own group. |

```json
{
  "success": true,
  "symbol": "WIPRO",
  "data": {
    "company_name": "Wipro Ltd",
    "screener_slug": "WIPRO",
    "screener_url": "https://www.screener.in/company/WIPRO/consolidated/",
    "source": "screener.in",
    "annual_reports_requested": 1,
    "concall_years": 3,
    "latest_annual_report": "https://www.bseindia.com/...pdf",
    "annual_reports": [
      {"financial_year": "FY2026", "url": "https://www.bseindia.com/...pdf", "source": "bse"}
    ],
    "concalls": [
      {"date": "2026-07", "period": "Jul 2026",
       "transcript_url": "https://www.bseindia.com/...pdf", "ppt_url": ""}
    ],
    "counts": {
      "annual_reports_found": 1, "annual_reports_available": 15,
      "concalls_found": 13, "concalls_available": 39,
      "transcripts_found": 13, "ppts_found": 6
    }
  }
}
```

Consumer notes:

- Every returned annual report carries a URL. Entries are **not** padded to
  the requested count, so a company with fewer published reports than
  requested returns fewer entries. Compare `counts.annual_reports_found`
  against `counts.annual_reports_available` to see whether more exist.
- `latest_annual_report` is a convenience copy of the newest URL, since
  fetching it is the most common reason to call this endpoint. It is an
  empty string when the company has no discoverable report.
- A transcript label is not a guarantee of a transcript link. Screener
  renders the label for some companies as an inert element with no href at
  all — every one of ITC's 37 and Trent's 14 concalls, for instance — so
  `transcript_url` is legitimately empty there and the document does not
  exist to be fetched.
- `period` is **not** a unique key. A single quarter can appear twice when
  both an exchange-hosted and a company-IR-hosted document exist for the
  same call, so deduplicate downstream by document content hash.
- Empty-string URLs mean "not published / not found", not an error.
- Linked PDFs are directly downloadable; BSE `AnnPdfOpen.aspx` links need
  no Referer header or cookie.

---

### 3. Institutional Ratio Analytics Endpoints

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
└── cashflow.json       # Operating cash flows, CapEx, and Free Cash Flow
```

The collector service persists **statement JSON only**. Filing PDFs are not
stored there — `/documents` returns links, and the GrowNXT ingestion layer
owns downloading and persisting them under its own
`data/<ticker>/documents/` tree. Note also that the deployed service writes
to an ephemeral serverless path (`/tmp/data`), so it must never be treated
as a durable document store.
