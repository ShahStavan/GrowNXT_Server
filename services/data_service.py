"""Data Service Handler for Stock Financial Statements.

Ingests fundamental financial statement datasets from the Financial Data Collector
Vercel REST API service (https://financial-data-collector-qrxj.vercel.app).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL

logger = logging.getLogger(__name__)


class StockDataHandler:
    """Handle fetching and saving stock financial data from Vercel REST service."""
    
    def __init__(self, output_dir: str):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base_url = FINANCIAL_DATA_COLLECTOR_BASE_URL.rstrip('/')

    def _save(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as pretty-printed JSON."""
        with open(path, 'w', encoding='utf-8') as file_handle:
            json.dump(data, file_handle, ensure_ascii=False, indent=2)

    def _load(self, path: Path) -> Dict[str, Any]:
        """Load JSON data from disk."""
        with open(path, 'r', encoding='utf-8') as file_handle:
            return json.load(file_handle)

    def fetch_stock_financials(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch aggregated financial statement dataset for symbol from Vercel API."""
        url = f"{self.base_url}/api/v1/stocks/{symbol.lower()}/financials"
        try:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data.get("data", {})
            logger.warning("Vercel API returned error for %s: %s", symbol, data.get("message"))
            return None
        except Exception as exc:
            logger.error("Failed fetching financials for %s from Vercel API: %s", symbol, exc)
            return None

    def save_all(self, folder: Path, sid: str, symbol: Optional[str] = None) -> bool:
        """Fetch complete financial statement bundle from Vercel API and save to local disk."""
        ticker_symbol = symbol or folder.name
        financials = self.fetch_stock_financials(ticker_symbol)

        if not financials:
            logger.error("Unable to save data for %s: Vercel API response empty", ticker_symbol)
            return False

        try:
            folder.mkdir(parents=True, exist_ok=True)

            if "quarterly" in financials:
                self._save({"quarterlyData": financials["quarterly"]}, folder / "quarterly.json")

            if "annual" in financials:
                self._save({"annualData": financials["annual"]}, folder / "annual.json")

            if "balancesheet" in financials:
                self._save({"balancesheetData": financials["balancesheet"]}, folder / "balancesheet.json")

            if "cashflow" in financials:
                self._save({"cashflowData": financials["cashflow"]}, folder / "cashflow.json")

            if "summary" in financials:
                self._save(financials["summary"], folder / "summary.json")

            logger.info("Successfully saved financial statement bundle for %s -> %s", ticker_symbol.upper(), folder)
            return True

        except Exception as exc:
            logger.error("Failed writing stock statement files for %s: %s", ticker_symbol, exc)
            return False
