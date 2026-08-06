"""Financial Data Service Connector Script.

Replaces legacy Selenium scrapers by connecting directly to the live Financial
Data Collector Vercel REST service:
https://financial-data-collector-qrxj.vercel.app
"""

import logging
import sys
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL
from services.data_service import StockDataHandler
from api.search import StockSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def process_stock_ticker(symbol: str, output_dir: str = "D:/Stock_Fundamental/data") -> bool:
    """Fetches and saves full financial statement bundle from Vercel service."""
    searcher = StockSearch(Path(output_dir))
    matches = searcher.search_stock(symbol)
    if not matches:
        logger.error("No stock matches found for query: %s", symbol)
        return False
    
    target_item = matches[0]
    logger.info("Found stock match: %s (%s)", target_item.get('name'), target_item.get('ticker'))
    return searcher.save_stock_data(target_item)


if __name__ == "__main__":
    ticker_input = input("Enter stock ticker symbol (e.g. WIPRO, ADANIENT): ").strip()
    if ticker_input:
        success = process_stock_ticker(ticker_input)
        if success:
            print(f"✅ Data for {ticker_input.upper()} successfully saved!")
        else:
            print(f"❌ Data ingestion failed for {ticker_input.upper()}")
