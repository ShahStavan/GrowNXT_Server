"""Batch Financial Dataset Ingestion Script for NIFTY 50 Tickers.

Sequentially ingests fundamental financial statement datasets for constituent
stocks of the NIFTY 50 index with rate-limiting delay buffers.

Google Python Style Guide Compliant.
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from api.search import StockSearch
from core.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Complete List of Constituent NIFTY 50 Tickers
NIFTY_50: List[str] = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDIGO", "INFY", "ITC", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TMPV", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN",
]


def _find_match(results: List[Dict[str, Any]], ticker: str) -> Optional[Dict[str, Any]]:
    """Locates exact or primary ticker match in search results list.

    Args:
        results (List[Dict[str, Any]]): List of stock search result dicts.
        ticker (str): Target stock ticker string.

    Returns:
        Optional[Dict[str, Any]]: Exact or default top match dictionary.
    """
    for record in results:
        if record.get("ticker") == ticker:
            return record
    return results[0] if results else None


def fetch_all_nifty50(output_directory: Optional[Path] = None, delay_seconds: float = 1.5) -> None:
    """Ingests financial statement bundles for all NIFTY 50 tickers.

    Args:
        output_directory (Optional[Path]): Directory path for storing ingested data.
        delay_seconds (float): Rate-limiting pause between API requests. Defaults to 1.5s.
    """
    target_dir = Path(output_directory) if output_directory else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    searcher = StockSearch(target_dir)
    successful_tickers: List[str] = []
    failed_tickers: List[str] = []
    missing_tickers: List[str] = []

    total_count = len(NIFTY_50)
    logger.info("Initiating batch ingestion for %d NIFTY 50 stocks...", total_count)

    for idx, ticker in enumerate(NIFTY_50, 1):
        try:
            results = searcher.search_stock(ticker)
            if not results:
                missing_tickers.append(ticker)
                logger.warning("[%d/%d] X %s - Stock record not found", idx, total_count, ticker)
                time.sleep(0.5)
                continue

            match = _find_match(results, ticker)
            if not match:
                missing_tickers.append(ticker)
                logger.warning("[%d/%d] X %s - No candidate match in search results", idx, total_count, ticker)
                continue

            save_ok = searcher.save_stock_data(match)
            if save_ok:
                successful_tickers.append(ticker)
                logger.info("[%d/%d] V %s - Successfully ingested", idx, total_count, ticker)
            else:
                failed_tickers.append(ticker)
                logger.error("[%d/%d] X %s - Ingestion save failed", idx, total_count, ticker)

            time.sleep(delay_seconds)

        except Exception as exc:
            failed_tickers.append(ticker)
            logger.error("[%d/%d] X %s - Exception encountered: %s", idx, total_count, ticker, exc, exc_info=True)

    # Output Summary
    logger.info("=" * 60)
    logger.info(
        "Ingestion Batch Complete | Total: %d | Successful: %d | Failed: %d | Missing: %d",
        total_count,
        len(successful_tickers),
        len(failed_tickers),
        len(missing_tickers),
    )
    if failed_tickers:
        logger.warning("Failed Tickers: %s", ", ".join(failed_tickers))
    if missing_tickers:
        logger.warning("Missing Tickers: %s", ", ".join(missing_tickers))


if __name__ == "__main__":
    fetch_all_nifty50()
