"""LangChain Financial Data Tools and Smart Context Retrieval Agent.

Integrates with the live Financial Data Collector Vercel REST Service to expose
standalone LangChain `@tool` definitions and an intelligent context retrieval agent.

Vercel Endpoints Supported:
- /api/v1/stocks/search
- /api/v1/stocks/<symbol>/income/quarterly
- /api/v1/stocks/<symbol>/income/annual
- /api/v1/stocks/<symbol>/income/quarterly/growth
- /api/v1/stocks/<symbol>/income/annual/growth
- /api/v1/stocks/<symbol>/balancesheet
- /api/v1/stocks/<symbol>/balancesheet/growth
- /api/v1/stocks/<symbol>/cashflow
- /api/v1/stocks/<symbol>/summary
- /api/v1/stocks/<symbol>/peers
- /api/v1/stocks/<symbol>/dupont
- /api/v1/stocks/<symbol>/solvency
- /api/v1/stocks/<symbol>/liquidity
- /api/v1/stocks/<symbol>/capital-efficiency
- /api/v1/stocks/<symbol>/cagr
- /api/v1/stocks/<symbol>/financials

Google Python Style Guide Compliant.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
import requests

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL

logger = logging.getLogger(__name__)

# Base Service Endpoint
SERVICE_BASE_URL: str = os.getenv(
    "FINANCIAL_DATA_SERVICE_URL",
    FINANCIAL_DATA_COLLECTOR_BASE_URL
).rstrip("/")


def _http_get_json(endpoint_path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Helper function to perform HTTP GET requests to the Financial Data Collector service.

    Args:
        endpoint_path (str): Relative API endpoint URL string.
        params (Optional[Dict[str, Any]]): Query parameters.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary response, or None on failure.
    """
    url = f"{SERVICE_BASE_URL}{endpoint_path}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error("HTTP GET request failed for endpoint %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected exception during HTTP GET to %s: %s", url, exc, exc_info=True)
        return None


# -----------------------------------------------------------------------------
# LangChain @tool Definitions
# -----------------------------------------------------------------------------

@tool
def search_stock_ticker_tool(query: str) -> str:
    """Searches for stock ticker symbols and Stock Identifiers (SIDs) given a company query.

    Args:
        query (str): Company name or symbol fragment.

    Returns:
        str: JSON string of matching stock records.
    """
    data = _http_get_json("/api/v1/stocks/search", params={"q": query})
    if not data or not data.get("success"):
        return f"No ticker search results found for query: '{query}'"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_quarterly_income_tool(symbol: str) -> str:
    """Fetches 8-quarter interim income statement (Quarterly Sales, Operating Profit, PAT, EPS).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Quarterly income statement JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/quarterly")
    if not data or not data.get("success"):
        return f"Quarterly income statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_annual_income_tool(symbol: str) -> str:
    """Fetches 5-year annual income statement (Total Revenue, Operating Profit EBIT, PAT, EPS).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Annual income statement JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/annual")
    if not data or not data.get("success"):
        return f"Annual income statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_quarterly_growth_tool(symbol: str) -> str:
    """Fetches QoQ quarterly growth metrics with plain-English annotations.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Quarterly growth JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/quarterly/growth")
    if not data or not data.get("success"):
        return f"Quarterly growth metrics unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_annual_growth_tool(symbol: str) -> str:
    """Fetches YoY annual growth metrics with plain-English annotations.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Annual growth JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/income/annual/growth")
    if not data or not data.get("success"):
        return f"Annual growth metrics unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_balance_sheet_tool(symbol: str) -> str:
    """Fetches annual balance sheet statements (Net Worth Equity, Total Debt, Bank Cash).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Balance sheet JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/balancesheet")
    if not data or not data.get("success"):
        return f"Balance sheet statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_balancesheet_growth_tool(symbol: str) -> str:
    """Fetches balance sheet solvency growth metrics with plain-English annotations.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Balance sheet growth JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/balancesheet/growth")
    if not data or not data.get("success"):
        return f"Balance sheet growth metrics unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_cash_flow_tool(symbol: str) -> str:
    """Fetches annual cash flow statements (Operating Cash Flow CFO, CapEx, Free Cash Flow FCF).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Cash flow statement JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/cashflow")
    if not data or not data.get("success"):
        return f"Cash flow statement unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_stock_summary_tool(symbol: str) -> str:
    """Fetches stock summary profile, business description, and market cap.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Stock profile summary JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/summary")
    if not data or not data.get("success"):
        return f"Stock summary profile unavailable for stock: {symbol}"
    return json.dumps(data.get("data", {}), indent=2)


@tool
def fetch_stock_peers_tool(symbol: str) -> str:
    """Fetches list of peer companies with tickers, names, and SIDs.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Peer companies list JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/peers")
    if not data or not data.get("success"):
        return f"Peer company list unavailable for stock: {symbol}"
    return json.dumps(data.get("data", []), indent=2)


@tool
def fetch_dupont_analysis_tool(symbol: str) -> str:
    """Fetches Extended 5-Factor DuPont ROE Model (Tax Burden, Interest Burden, Operating Margin %, Asset Turnover x, Equity Multiplier x).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: DuPont analysis JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/dupont")
    if not data or not data.get("success"):
        return f"DuPont analysis data unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_solvency_metrics_tool(symbol: str) -> str:
    """Fetches Solvency & Coverage: Interest Coverage Ratio (ICR), Net Debt, Net Debt-to-EBITDA x, Debt-to-Equity x.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Solvency metrics JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/solvency")
    if not data or not data.get("success"):
        return f"Solvency metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_liquidity_metrics_tool(symbol: str) -> str:
    """Fetches Working Capital Health: Current Ratio x, Quick Ratio x, Receivable Days (DSO), Inventory Days (DIO).

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Liquidity metrics JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/liquidity")
    if not data or not data.get("success"):
        return f"Liquidity metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_capital_efficiency_tool(symbol: str) -> str:
    """Fetches Capital Allocation: ROIC %, Free Cash Flow (FCF) Conversion %, Fixed Asset Turnover x.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Capital efficiency JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/capital-efficiency")
    if not data or not data.get("success"):
        return f"Capital efficiency metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_cagr_metrics_tool(symbol: str) -> str:
    """Fetches Multi-Year Growth: 3-Year & 5-Year Revenue CAGR %, EBIT CAGR %, PAT CAGR %.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: CAGR metrics JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/cagr")
    if not data or not data.get("success"):
        return f"CAGR metrics unavailable for stock: {symbol}"
    return json.dumps(data, indent=2)


@tool
def fetch_full_financial_bundle_tool(symbol: str) -> str:
    """Fetches complete aggregated financial statement dataset for a stock.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        str: Aggregated financial bundle JSON payload.
    """
    data = _http_get_json(f"/api/v1/stocks/{symbol.lower()}/financials")
    if not data or not data.get("success"):
        return f"Full financial bundle unavailable for stock: {symbol}"
    return json.dumps(data.get("data", {}), indent=2)


# Exported Tools List Registry
ALL_FINANCIAL_TOOLS = [
    search_stock_ticker_tool,
    fetch_quarterly_income_tool,
    fetch_annual_income_tool,
    fetch_quarterly_growth_tool,
    fetch_annual_growth_tool,
    fetch_balance_sheet_tool,
    fetch_balancesheet_growth_tool,
    fetch_cash_flow_tool,
    fetch_stock_summary_tool,
    fetch_stock_peers_tool,
    fetch_dupont_analysis_tool,
    fetch_solvency_metrics_tool,
    fetch_liquidity_metrics_tool,
    fetch_capital_efficiency_tool,
    fetch_cagr_metrics_tool,
    fetch_full_financial_bundle_tool,
]


class FinancialDataAgent:
    """Smart Context Agent that dynamically invokes Vercel REST tools based on report requirements.

    Args:
        service_url (Optional[str]): Service URL override.
    """

    def __init__(self, service_url: Optional[str] = None) -> None:
        self.service_url: str = service_url or SERVICE_BASE_URL

    def get_context_for_section(self, section_key: str, symbol: str) -> str:
        """Dynamically selects API endpoint tools and fetches JSON metrics live for section prompt metadata.

        Args:
            section_key (str): Section identifier (e.g., 'dupont_analysis', 'solvency_analysis').
            symbol (str): Stock ticker symbol (e.g., 'ADANIENT', 'WIPRO').

        Returns:
            str: Formatted context string injected into LLM prompt context.
        """
        symbol_upper = symbol.upper()
        logger.info("FinancialDataAgent executing dynamic tool retrieval for section '%s' on symbol '%s'", section_key, symbol_upper)

        context_parts: List[str] = []

        if section_key == "dupont_analysis":
            dupont_json = fetch_dupont_analysis_tool.invoke({"symbol": symbol_upper})
            cagr_json = fetch_cagr_metrics_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- EXTENDED DUPONT ROE MODEL ---\n{dupont_json}")
            context_parts.append(f"--- MULTI-YEAR COMPOUND ANNUAL GROWTH RATES (CAGR) ---\n{cagr_json}")

        elif section_key == "solvency_analysis":
            solvency_json = fetch_solvency_metrics_tool.invoke({"symbol": symbol_upper})
            bal_growth_json = fetch_balancesheet_growth_tool.invoke({"symbol": symbol_upper})
            balance_json = fetch_balance_sheet_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- SOLVENCY & COVERAGE METRICS ---\n{solvency_json}")
            context_parts.append(f"--- BALANCE SHEET SOLVENCY GROWTH ---\n{bal_growth_json}")
            context_parts.append(f"--- BALANCE SHEET STATEMENT ---\n{balance_json}")

        elif section_key == "financial_performance":
            q_json = fetch_quarterly_income_tool.invoke({"symbol": symbol_upper})
            q_growth_json = fetch_quarterly_growth_tool.invoke({"symbol": symbol_upper})
            a_json = fetch_annual_income_tool.invoke({"symbol": symbol_upper})
            a_growth_json = fetch_annual_growth_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- 8-QUARTER INTERIM INCOME STATEMENT ---\n{q_json}")
            context_parts.append(f"--- QUARTERLY QoQ GROWTH METRICS ---\n{q_growth_json}")
            context_parts.append(f"--- 5-YEAR ANNUAL INCOME STATEMENT ---\n{a_json}")
            context_parts.append(f"--- ANNUAL YoY GROWTH METRICS ---\n{a_growth_json}")

        elif section_key in ("core_business", "market_footprint", "executive_summary"):
            summary_json = fetch_stock_summary_tool.invoke({"symbol": symbol_upper})
            peers_json = fetch_stock_peers_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- STOCK SUMMARY PROFILE & BUSINESS OVERVIEW ---\n{summary_json}")
            context_parts.append(f"--- PEER COMPANIES LIST ---\n{peers_json}")

        elif section_key in ("expansion_plans", "investment_thesis"):
            cap_json = fetch_capital_efficiency_tool.invoke({"symbol": symbol_upper})
            liq_json = fetch_liquidity_metrics_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- CAPITAL ALLOCATION & ROIC ---\n{cap_json}")
            context_parts.append(f"--- WORKING CAPITAL HEALTH & LIQUIDITY ---\n{liq_json}")

        else:
            bundle_json = fetch_full_financial_bundle_tool.invoke({"symbol": symbol_upper})
            context_parts.append(f"--- FULL FINANCIAL STATEMENT BUNDLE ---\n{bundle_json}")

        return "\n\n".join(context_parts)
