"""LangChain Financial Data Tools and Smart Context Retrieval Agent.

Integrates with the live Financial Data Collector Vercel REST Service:
https://financial-data-collector-qrxj.vercel.app

Provides standalone LangChain @tool definitions and an intelligent retrieval
agent that dynamically selects and executes the appropriate API tool based
on report section requirements.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Base Service Endpoint
SERVICE_BASE_URL = os.getenv(
    "FINANCIAL_DATA_SERVICE_URL",
    "https://financial-data-collector-qrxj.vercel.app"
).rstrip("/")


def _http_get_json(endpoint_path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Helper function to perform HTTP GET requests to the Financial Data Collector service."""
    url = f"{SERVICE_BASE_URL}{endpoint_path}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error("HTTP GET failed for endpoint %s: %s", url, exc)
        return None


@tool
def search_stock_ticker_tool(query: str) -> str:
    """Searches for stock ticker symbols and Stock Identifiers (SIDs) given a company name or query string."""
    data = _http_get_json("/api/v1/stocks/search", params={"q": query})
    if not data or not data.get("success"):
        return f"No ticker search results found for query: '{query}'"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_dupont_analysis_tool(symbol: str) -> str:
    """Fetches ground-truth Extended 5-Factor DuPont ROE breakdown (Tax Burden, Interest Burden, Operating Margin, Asset Turnover, Leverage)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/dupont")
    if not data or not data.get("success"):
        return f"DuPont analysis data unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_solvency_metrics_tool(symbol: str) -> str:
    """Fetches Solvency & Debt Coverage metrics (Interest Coverage Ratio, Net Debt, Net Debt/EBITDA, Debt-to-Equity)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/solvency")
    if not data or not data.get("success"):
        return f"Solvency metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_liquidity_metrics_tool(symbol: str) -> str:
    """Fetches Liquidity & Working Capital Health metrics (Current Ratio, Quick Ratio, Receivable Days DSO, Inventory Days DIO)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/liquidity")
    if not data or not data.get("success"):
        return f"Liquidity metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_capital_efficiency_tool(symbol: str) -> str:
    """Fetches Capital Allocation & Efficiency metrics (ROIC %, Free Cash Flow Conversion %, Fixed Asset Turnover)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/capital-efficiency")
    if not data or not data.get("success"):
        return f"Capital efficiency metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_cagr_metrics_tool(symbol: str) -> str:
    """Fetches Multi-Year Compound Annual Growth Rates (3-Year and 5-Year Revenue, EBIT, and Net Profit PAT CAGR)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/cagr")
    if not data or not data.get("success"):
        return f"CAGR metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_quarterly_income_tool(symbol: str) -> str:
    """Fetches 8-quarter interim income statement (Quarterly Sales, Operating Profit, PAT, EPS)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/quarterly")
    if not data or not data.get("success"):
        return f"Quarterly income statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_annual_income_tool(symbol: str) -> str:
    """Fetches 5-year annual income statement (Total Revenue, Operating Profit EBIT, PAT, EPS)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/annual")
    if not data or not data.get("success"):
        return f"Annual income statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_balance_sheet_tool(symbol: str) -> str:
    """Fetches annual balance sheet statements (Net Worth Equity, Total Debt, Bank Cash, Current Assets, Liabilities)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/balancesheet")
    if not data or not data.get("success"):
        return f"Balance sheet statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_cash_flow_tool(symbol: str) -> str:
    """Fetches annual cash flow statements (Operating Cash Flow CFO, CapEx, Free Cash Flow FCF, Financing CFF)."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/cashflow")
    if not data or not data.get("success"):
        return f"Cash flow statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_stock_summary_tool(symbol: str) -> str:
    """Fetches stock summary profile, business overview, market cap, and peer company list."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/summary")
    if not data or not data.get("success"):
        return f"Stock summary profile unavailable for stock: {symbol}"
    return json.dumps(data.get("data", {}), indent=2)


@tool
def fetch_full_financial_bundle_tool(symbol: str) -> str:
    """Fetches complete aggregated financial dataset (Income, Balance Sheet, Cash Flow, Growth, Summary) for a stock."""
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/financials")
    if not data or not data.get("success"):
        return f"Full financial bundle unavailable for stock: {symbol}"
    return json.dumps(data.get("data", {}), indent=2)


# Exported Tools Registry
ALL_FINANCIAL_TOOLS = [
    search_stock_ticker_tool,
    fetch_dupont_analysis_tool,
    fetch_solvency_metrics_tool,
    fetch_liquidity_metrics_tool,
    fetch_capital_efficiency_tool,
    fetch_cagr_metrics_tool,
    fetch_quarterly_income_tool,
    fetch_annual_income_tool,
    fetch_balance_sheet_tool,
    fetch_cash_flow_tool,
    fetch_stock_summary_tool,
    fetch_full_financial_bundle_tool
]


class FinancialDataAgent:
    """Smart Retrieval Agent that dynamically selects and executes LangChain tools for RAG report generation."""

    def __init__(self, service_url: Optional[str] = None) -> None:
        self.service_url = service_url or SERVICE_BASE_URL

    def get_context_for_section(self, section_key: str, symbol: str) -> str:
        """Dynamically selects tools and executes REST API calls based on section requirements.

        Args:
            section_key: Key identifying report section (e.g. 'dupont_analysis', 'solvency').
            symbol: Stock ticker symbol (e.g. 'ADANIENT', 'WIPRO').

        Returns:
            Formatted JSON string payload injected dynamically into LLM prompt context.
        """
        symbol_upper = symbol.upper()
        logger.info("FinancialDataAgent executing dynamic retrieval for section '%s' on %s", section_key, symbol_upper)

        context_parts = []

        if section_key == "dupont_analysis":
            dupont_json = fetch_dupont_analysis_tool.invoke({"symbol": symbol_upper})
            cagr_json = fetch_cagr_metrics_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- LIVE EXTENDED DUPONT ROE ANALYSIS ---\n{dupont_json}")
            context_parts.append(f"--- MULTI-YEAR COMPOUND ANNUAL GROWTH RATES (CAGR) ---\n{cagr_json}")

        elif section_key == "solvency_analysis":
            solvency_json = fetch_solvency_metrics_tool.invoke({"symbol": symbol_upper})
            balance_json = fetch_balance_sheet_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- LIVE SOLVENCY & DEBT COVERAGE METRICS ---\n{solvency_json}")
            context_parts.append(f"--- BALANCE SHEET STATEMENT ---\n{balance_json}")

        elif section_key == "financial_performance":
            q_json = fetch_quarterly_income_tool.invoke({"symbol": symbol_upper})
            a_json = fetch_annual_income_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- 8-QUARTER INTERIM INCOME STATEMENT ---\n{q_json}")
            context_parts.append(f"--- 5-YEAR ANNUAL INCOME STATEMENT ---\n{a_json}")

        elif section_key in ("core_business", "market_footprint", "executive_summary"):
            summary_json = fetch_stock_summary_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- STOCK PROFILE & BUSINESS OVERVIEW ---\n{summary_json}")

        elif section_key in ("expansion_plans", "investment_thesis"):
            cap_json = fetch_capital_efficiency_tool.invoke({"symbol": symbol_upper})
            liq_json = fetch_liquidity_metrics_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- CAPITAL EFFICIENCY & ROIC ---\n{cap_json}")
            context_parts.append(f"--- WORKING CAPITAL LIQUIDITY ---\n{liq_json}")

        else:
            bundle_json = fetch_full_financial_bundle_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- FULL FINANCIAL STATEMENT BUNDLE ---\n{bundle_json}")

        return "\n\n".join(context_parts)
