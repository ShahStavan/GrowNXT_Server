"""Stock Search and Ticker Resolution API.

Uses Financial Data Collector Vercel REST service to search tickers and save
statement datasets locally.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL
from services.data_service import StockDataHandler
from services.enrichment_service import CompanyEnricher, JSONEncoder

logger = logging.getLogger(__name__)


class StockSearch:
    """Handles searching stock tickers and triggering local data ingestion."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.base_url = FINANCIAL_DATA_COLLECTOR_BASE_URL.rstrip('/')
        self.search_url = f"{self.base_url}/api/v1/stocks/search"
        self.enricher = CompanyEnricher(output_dir)

    def search_stock(self, q: str) -> List[Dict[str, Any]]:
        """Queries stock discovery endpoint."""
        query_str = q.strip() if q else ""
        if not query_str:
            return []
        
        try:
            res = requests.get(
                self.search_url,
                params={"q": query_str},
                timeout=15
            )
            res.raise_for_status()
            data = res.json()
            
            if not data.get('success'):
                logger.warning("Stock search returned error for '%s': %s", query_str, data.get('message'))
                return []
            
            return data.get('data', [])
            
        except Exception as exc:
            logger.error("Stock search request failed for '%s': %s", query_str, exc)
            return []

    def instant_search(self, q: str) -> List[Dict[str, Any]]:
        """Performs search and logs matching results."""
        items = self.search_stock(q)
        self._display(items)
        return items

    def _display(self, items: List[Dict[str, Any]]) -> None:
        """Prints matched stocks."""
        if not items:
            print("No stocks found")
            return
        
        print("\nResults:")
        for i, item in enumerate(items, 1):
            print(f"{i}. {item.get('name')} ({item.get('ticker')})")
    
    def save_stock_data(self, item: Dict[str, Any]) -> bool:
        """Saves stock metadata and fetches financial statement bundle from Vercel API."""
        ticker = item.get('ticker')
        if not ticker:
            return False
        
        folder = self.output_dir / ticker.lower()
        folder.mkdir(parents=True, exist_ok=True)
        
        # Skip if complete and enriched
        if self._check_exists(folder):
            logger.info("Stock data for %s already exists and enriched, skipping fetch", ticker)
            return True
        
        # Save basic metadata
        data = {
            'ticker': ticker,
            'sid': item.get('sid', ticker[:4]),
            'name': item.get('name'),
            'sector': item.get('sector'),
            'brands': item.get('brands', []),
            'marketCap': item.get('marketCap'),
            'quote': item.get('quote', {}),
            'slug': item.get('slug')
        }
        
        with open(folder / 'sData.json', 'w', encoding='utf-8') as file_handle:
            json.dump(data, file_handle, indent=2, cls=JSONEncoder, ensure_ascii=False)

        # Fetch financial statement bundle via Vercel REST service
        handler = StockDataHandler(str(self.output_dir))
        ok = handler.save_all(folder, data['sid'], symbol=ticker)
        
        # Enrich data
        try:
            if self.enricher.enrich(ticker):
                logger.info("Successfully enriched company brief for %s", ticker)
            else:
                logger.warning("Company brief enrichment failed for %s", ticker)
        except Exception as exc:
            logger.error("Enrichment exception for %s: %s", ticker, exc)
        
        return ok
    
    def _check_exists(self, folder: Path) -> bool:
        """Check if all files exist and sData has company brief enrichment."""
        files = ['sData.json', 'quarterly.json', 'annual.json', 'balancesheet.json', 'cashflow.json']
        
        for file_basename in files:
            if not (folder / file_basename).exists():
                return False
        
        try:
            with open(folder / 'sData.json', 'r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)
                return 'company_brief' in data and bool(data.get('company_brief'))
        except Exception:
            return False