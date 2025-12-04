import requests
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from services.data_service import StockDataHandler
from core.config import API_ENDPOINTS, HTTP_HEADERS
from services.enrichment_service import CompanyEnricher, JSONEncoder

class StockSearch:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.search_url = API_ENDPOINTS.SEARCH
        self.headers = HTTP_HEADERS
        self.enricher = CompanyEnricher(output_dir)

    def search_stock(self, q: str) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        
        try:
            res = requests.get(
                self.search_url,
                params={"text": q.strip(), "types": "stock", "pageNumber": 0},
                headers=self.headers,
                timeout=10
            )
            res.raise_for_status()
            data = res.json()
            
            if not data.get('success'):
                print(f"API error: {data.get('message')}")
                return []
            
            return data.get('data', {}).get('items', [])
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def instant_search(self, q: str) -> List[Dict[str, Any]]:
        items = self.search_stock(q)
        self._display(items)
        return items

    def _display(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            print("No stocks found")
            return
        
        print("\nResults:")
        for i, item in enumerate(items, 1):
            print(f"{i}. {item.get('name')} ({item.get('ticker')})")
    
    def save_stock_data(self, item: Dict[str, Any]) -> bool:
        ticker = item.get('ticker')
        if not ticker:
            return False
        
        folder = self.output_dir / ticker.lower()
        folder.mkdir(exist_ok=True)
        
        # Skip if complete
        if self._check_exists(folder):
            print(f"✓ {ticker} exists, skipping")
            return True
        
        # Save basic data
        data = {
            'ticker': ticker,
            'sid': item.get('sid'),
            'name': item.get('name'),
            'sector': item.get('sector'),
            'brands': item.get('brands', []),
            'marketCap': item.get('marketCap'),
            'quote': item.get('quote', {}),
            'slug': item.get('slug')
        }
        
        with open(folder / 'sData.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=JSONEncoder, ensure_ascii=False)

        # Fetch financial data
        handler = StockDataHandler(str(self.output_dir))
        ok = handler.save_all(folder, data['sid'])
        
        # Enrich data
        try:
            if self.enricher.enrich(ticker):
                print(f"✓ {ticker} enriched")
            else:
                print(f"⚠ {ticker} enrich failed")
        except Exception as e:
            print(f"✗ {ticker} enrich error: {e}")
        
        return ok
    
    def _check_exists(self, folder: Path) -> bool:
        """Check if all files exist and sData has enrichment"""
        files = ['sData.json', 'quarterly.json', 'annual.json', 'balancesheet.json', 'cashflow.json']
        
        for f in files:
            if not (folder / f).exists():
                return False
        
        try:
            with open(folder / 'sData.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                return 'company_brief' in data and data.get('company_brief')
        except Exception:
            return False
    
    def process_search(self, q: str) -> Optional[Dict[str, Any]]:
        items = self.search_stock(q)
        self._display(items)
        
        if not items:
            return None
        
        try:
            sel = int(input("\nSelect (0 to cancel): "))
            if 1 <= sel <= len(items):
                item = items[sel - 1]
                if self.save_stock_data(item):
                    return item
        except ValueError:
            print("Invalid")
        
        return None