from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

# File paths
class FILE_PATHS:
    STOCK_LISTINGS = DATA_DIR / "stock_listings.xlsx"
    STOCK_DATA = DATA_DIR / "stockData.xlsx"
    FILTERED_STOCK_LISTINGS = DATA_DIR / "filtered_stock_listings.xlsx"

# Website URLs
class WEBSITE_URLS:
    TICKERTAPE_STOCKS = "https://www.tickertape.in/stocks?filter="
    SCREENER_API_SEARCH = "https://www.screener.in/api/company/search/"
    SCREENER_BASE = "https://www.screener.in"

# API URLs
class API_ENDPOINTS:
    BASE_URL = "https://api.tickertape.in"
    ANALYZE_BASE_URL = "https://analyze.api.tickertape.in"
    SEARCH = f"{BASE_URL}/search"
    QUARTERLY = f"{BASE_URL}/stocks/financials/income/{{sid}}/interim/normal"
    ANNUAL = f"{BASE_URL}/stocks/financials/income/{{sid}}/annual/normal"
    QT_GROWTH = f"{BASE_URL}/stocks/financials/income/{{sid}}/interim/growth"
    AN_GROWTH = f"{BASE_URL}/stocks/financials/income/{{sid}}/annual/growth"
    BALANCE_SHEET = f"{BASE_URL}/stocks/financials/balancesheet/{{sid}}/annual/normal"
    BAL_GROWTH = f"{BASE_URL}/stocks/financials/balancesheet/{{sid}}/annual/growth"
    CASH_FLOW = f"{BASE_URL}/stocks/financials/cashflow/{{sid}}/annual/normal"
    SUMMARY = f"{ANALYZE_BASE_URL}/v2/stocks/summary/{{sid}}"

# Common HTTP Headers
HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Version': '8.14.0',
    'Origin': 'https://www.tickertape.in',
    'Referer': 'https://www.tickertape.in/',
    'Cookie': 'jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',  # Add your JWT token
    'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'x-csrf-token': '7610d2a1'
}

# AI Agent Requirements
REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]
