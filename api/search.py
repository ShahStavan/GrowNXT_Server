"""Stock Discovery and Ticker Resolution API Integration.

Connects to the Financial Data Collector Vercel REST Service to discover stock tickers,
verify symbol listings, and retrieve live financial metadata.

Google Python Style Guide Compliant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL

logger = logging.getLogger(__name__)


class StockSearch:
    """Handles stock ticker discovery via the Financial Data Collector Vercel REST Service.

    Args:
        output_dir (Optional[Path]): Base directory path for stock data folders.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir: Path = Path(output_dir) if output_dir else Path("./data")
        self.base_url: str = FINANCIAL_DATA_COLLECTOR_BASE_URL.rstrip("/")
        self.search_url: str = f"{self.base_url}/api/v1/stocks/search"

    def search_stock(self, query: str) -> List[Dict[str, Any]]:
        """Queries the external Vercel stock search endpoint.

        Args:
            query (str): Company name or ticker query string.

        Returns:
            List[Dict[str, Any]]: List of matching stock search result dictionaries.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return []

        try:
            response = requests.get(
                self.search_url,
                params={"q": cleaned_query},
                timeout=15
            )
            response.raise_for_status()
            payload = response.json()

            if not payload.get("success"):
                logger.warning("Stock search service returned error for '%s': %s", cleaned_query, payload.get("message"))
                return []

            return payload.get("data", [])

        except requests.RequestException as exc:
            logger.error("HTTP request error during stock search for '%s': %s", cleaned_query, exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error during stock search for '%s': %s", cleaned_query, exc, exc_info=True)
            return []

    def instant_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs search and logs search result summaries.

        Args:
            query (str): Search query.

        Returns:
            List[Dict[str, Any]]: Matched stock items.
        """
        results = self.search_stock(query)
        self._log_search_results(results)
        return results

    def _log_search_results(self, items: List[Dict[str, Any]]) -> None:
        """Logs matched stock results cleanly.

        Args:
            items (List[Dict[str, Any]]): List of stock metadata items.
        """
        if not items:
            logger.info("Stock search query returned no matching results.")
            return

        logger.info("Stock search matched %d items:", len(items))
        for index, item in enumerate(items, 1):
            logger.info("  %d. %s (%s)", index, item.get("name"), item.get("ticker"))

    def save_stock_data(self, item: Dict[str, Any]) -> bool:
        """Verifies stock ticker validity and ensures target directory exists.

        Args:
            item (Dict[str, Any]): Stock metadata payload dictionary.

        Returns:
            bool: True if symbol validation passes.
        """
        ticker = item.get("ticker")
        if not ticker:
            logger.error("Cannot process stock: missing 'ticker' attribute in payload.")
            return False

        ticker_folder = self.output_dir / ticker.lower()
        ticker_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Validated stock symbol '%s' directory at %s", ticker, ticker_folder)
        return True