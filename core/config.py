"""Centralized Configuration for GrowNXT Server."""

import os
from pathlib import Path

# Base System Paths
BASE_DIR = Path(r"D:/Stock_Fundamental")
OUTPUT_FOLDER = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"


# File paths
class FILE_PATHS:
    STOCK_LISTINGS = DATA_DIR / "stock_listings.xlsx"
    STOCK_DATA = DATA_DIR / "stockData.xlsx"
    FILTERED_STOCK_LISTINGS = DATA_DIR / "filtered_stock_listings.xlsx"


# Live Financial Data Collector Vercel REST Service URL
FINANCIAL_DATA_COLLECTOR_BASE_URL = os.getenv(
    "FINANCIAL_DATA_SERVICE_URL",
    "https://financial-data-collector-qrxj.vercel.app"
).rstrip("/")


# Website and Screener URLs
class WEBSITE_URLS:
    SCREENER_API_SEARCH = f"{FINANCIAL_DATA_COLLECTOR_BASE_URL}/api/v1/stocks/search"
    SCREENER_BASE = "https://www.screener.in"


# REST API Endpoint Templates
class API_ENDPOINTS:
    BASE_URL = FINANCIAL_DATA_COLLECTOR_BASE_URL
    SEARCH = f"{BASE_URL}/api/v1/stocks/search"
    QUARTERLY = f"{BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly"
    ANNUAL = f"{BASE_URL}/api/v1/stocks/{{sid}}/income/annual"
    QT_GROWTH = f"{BASE_URL}/api/v1/stocks/{{sid}}/income/quarterly/growth"
    AN_GROWTH = f"{BASE_URL}/api/v1/stocks/{{sid}}/income/annual/growth"
    BALANCE_SHEET = f"{BASE_URL}/api/v1/stocks/{{sid}}/balancesheet"
    BAL_GROWTH = f"{BASE_URL}/api/v1/stocks/{{sid}}/balancesheet/growth"
    CASH_FLOW = f"{BASE_URL}/api/v1/stocks/{{sid}}/cashflow"
    SUMMARY = f"{BASE_URL}/api/v1/stocks/{{sid}}/summary"
    DUPONT = f"{BASE_URL}/api/v1/stocks/{{sid}}/dupont"
    SOLVENCY = f"{BASE_URL}/api/v1/stocks/{{sid}}/solvency"
    LIQUIDITY = f"{BASE_URL}/api/v1/stocks/{{sid}}/liquidity"
    CAPITAL_EFFICIENCY = f"{BASE_URL}/api/v1/stocks/{{sid}}/capital-efficiency"
    CAGR = f"{BASE_URL}/api/v1/stocks/{{sid}}/cagr"


# Standard HTTP Headers
HTTP_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/134.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*'
}


# AI Agent Configuration
provider = os.getenv("LLM_PROVIDER", "gemini").lower()
if provider == "groq":
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
elif provider == "ollama":
    REQUIRED_ENV_VARS = []
else:
    REQUIRED_ENV_VARS = []