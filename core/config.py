"""Centralized System Configuration for GrowNXT Server.

This module provides system-wide constants, path management, REST API endpoints,
and environment setup definitions following Google Python Style Guide standards.

Senior Engineering Note:
    Avoid hardcoding absolute paths (e.g., 'D:/Stock_Fundamental'). Instead,
    we compute project-relative fallbacks and support environment overrides
    (GROWNXT_BASE_DIR, GROWNXT_DATA_DIR) for containerized and multi-platform deployments.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Final

# Project root directory derived dynamically relative to this file's position
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Base system paths with environment variable overrides for containerization support
_default_base_dir: Path = PROJECT_ROOT
env_base_dir: str = os.getenv("GROWNXT_BASE_DIR", "")
if env_base_dir:
    BASE_DIR: Final[Path] = Path(env_base_dir).resolve()
elif Path(r"D:/Stock_Fundamental").exists():
    # Legacy fallback for local dev environments where data was hosted externally
    BASE_DIR: Final[Path] = Path(r"D:/Stock_Fundamental")
else:
    BASE_DIR: Final[Path] = _default_base_dir

env_data_dir: str = os.getenv("GROWNXT_DATA_DIR", "")
DATA_DIR: Final[Path] = Path(env_data_dir).resolve() if env_data_dir else BASE_DIR / "data"

env_output_dir: str = os.getenv("GROWNXT_OUTPUT_DIR", "")
OUTPUT_FOLDER: Final[Path] = Path(env_output_dir).resolve() if env_output_dir else BASE_DIR / "output"

# Global Mapping file path (ground-truth metric dictionary)
MAPPING_FILE_PATH: Final[Path] = PROJECT_ROOT / "mapping.json"


@dataclass(frozen=True)
class FilePaths:
    """Immutable collection of core Excel and JSON dataset file paths."""

    stock_listings: Path = DATA_DIR / "stock_listings.xlsx"
    stock_data: Path = DATA_DIR / "stockData.xlsx"
    filtered_stock_listings: Path = DATA_DIR / "filtered_stock_listings.xlsx"
    mapping_json: Path = MAPPING_FILE_PATH

    # Legacy attribute alias support
    STOCK_LISTINGS: Path = DATA_DIR / "stock_listings.xlsx"
    STOCK_DATA: Path = DATA_DIR / "stockData.xlsx"
    FILTERED_STOCK_LISTINGS: Path = DATA_DIR / "filtered_stock_listings.xlsx"


FILE_PATHS: Final[FilePaths] = FilePaths()


# Live Financial Data Collector Vercel REST Service URL
FINANCIAL_DATA_COLLECTOR_BASE_URL: Final[str] = os.getenv(
    "FINANCIAL_DATA_SERVICE_URL",
    "https://financial-data-collector-qrxj.vercel.app"
).rstrip("/")


@dataclass(frozen=True)
class WebsiteUrls:
    """Web service and external API endpoint configurations."""

    screener_api_search: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/search"
    screener_base: str = "https://www.screener.in"

    # Legacy uppercase aliases
    SCREENER_API_SEARCH: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/search"
    SCREENER_BASE: str = "https://www.screener.in"


WEBSITE_URLS: Final[WebsiteUrls] = WebsiteUrls()


@dataclass(frozen=True)
class ApiEndpoints:
    """REST API endpoint paths for financial statement ingestion."""

    base_url: str = FINANCIAL_DATA_COLLECTOR_BASE_URL
    search: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/search"
    quarterly: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly"
    annual: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/annual"
    qt_growth: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly/growth"
    an_growth: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/annual/growth"
    balance_sheet: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/balancesheet"
    bal_growth: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/balancesheet/growth"
    cash_flow: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/cashflow"
    summary: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/summary"
    dupont: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/dupont"
    solvency: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/solvency"
    liquidity: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/liquidity"
    capital_efficiency: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/capital-efficiency"
    cagr: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/cagr"

    # Legacy uppercase aliases
    BASE_URL: str = FINANCIAL_DATA_COLLECTOR_BASE_URL
    SEARCH: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/search"
    QUARTERLY: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly"
    ANNUAL: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/annual"
    QT_GROWTH: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly/growth"
    AN_GROWTH: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/income/annual/growth"
    BALANCE_SHEET: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/balancesheet"
    BAL_GROWTH: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/balancesheet/growth"
    CASH_FLOW: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/cashflow"
    SUMMARY: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/summary"
    DUPONT: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/dupont"
    SOLVENCY: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/solvency"
    LIQUIDITY: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/liquidity"
    CAPITAL_EFFICIENCY: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/capital-efficiency"
    CAGR: str = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/{{sid}}/cagr"


API_ENDPOINTS: Final[ApiEndpoints] = ApiEndpoints()

# Standard HTTP Client Request Headers
HTTP_HEADERS: Final[Dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


def get_required_env_vars() -> List[str]:
    """Determines required environment variables dynamically based on configured LLM provider.

    Returns:
        List[str]: List of required environment variable names.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        return ["GROQ_API_KEY"]
    if provider == "gemini":
        return ["GEMINI_API_KEY"]
    return []


REQUIRED_ENV_VARS: List[str] = get_required_env_vars()