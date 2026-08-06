"""Financial Data Ingestion Command-Line Interface Script.

Connects to the Financial Data Collector Vercel REST Service to fetch and persist
the complete 7-part financial statement dataset bundle for any Indian stock ticker.

Google Python Style Guide Compliant.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

from api.search import StockSearch
from core.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def process_stock_ticker(symbol: str, output_directory: Optional[Path] = None) -> bool:
    """Fetches and persists full financial statement bundle for target ticker.

    Args:
        symbol (str): Target stock ticker symbol (e.g. 'WIPRO', 'ADANIENT').
        output_directory (Optional[Path]): Directory path for storing data. Defaults to DATA_DIR.

    Returns:
        bool: True if ingestion succeeded, False otherwise.
    """
    target_dir = Path(output_directory) if output_directory else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    searcher = StockSearch(target_dir)
    matches = searcher.search_stock(symbol)

    if not matches:
        logger.error("No stock ticker matches found for query: '%s'", symbol)
        return False

    target_item = matches[0]
    logger.info("Found stock match: %s (%s)", target_item.get("name"), target_item.get("ticker"))

    save_success = searcher.save_stock_data(target_item)
    if save_success:
        logger.info("Successfully ingested financial statement bundle for '%s'", symbol.upper())
    else:
        logger.error("Data ingestion failed for '%s'", symbol.upper())

    return save_success


def main() -> None:
    """CLI argument parser entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest stock financial statement bundle from REST service."
    )
    parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        help="Stock ticker symbol to fetch (e.g., WIPRO, ADANIENT, RELIANCE)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DATA_DIR),
        help="Output data directory path",
    )

    args = parser.parse_args()

    symbol_input = args.symbol
    if not symbol_input:
        symbol_input = input("Enter stock ticker symbol (e.g. WIPRO, ADANIENT): ").strip()

    if symbol_input:
        success = process_stock_ticker(symbol_input, Path(args.output_dir))
        sys.exit(0 if success else 1)
    else:
        logger.warning("No ticker symbol provided. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
