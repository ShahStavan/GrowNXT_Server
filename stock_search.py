import requests
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from data_handler import StockFundamentalData
from config import API_ENDPOINTS, HTTP_HEADERS
from company_agent import CompanyDataEnricher, JSONEncoder

class StockSearch:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.search_url = API_ENDPOINTS.SEARCH
        self.headers = HTTP_HEADERS
        self.data_enricher = CompanyDataEnricher(output_dir)

    def search_stock(self, query: str) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
            
        params = {
            "text": query.strip(),
            "types": "stock",
            "pageNumber": 0
        }
        
        try:
            response = requests.get(
                self.search_url, 
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 403:
                print("Access forbidden. Please check API permissions.")
                return []
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return []
                
            return data.get('data', {}).get('items', [])
            
        except requests.RequestException as e:
            print(f"Request failed: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {str(e)}")
            return []

    def instant_search(self, query: str) -> None:
        """Perform instant search and show results"""
        items = self.search_stock(query)
        self.display_results(items)
        return items

    def display_results(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            print("\nNo stocks found!")
            return
            
        print("\nSearch Results:")
        print("-" * 40)
        for idx, item in enumerate(items, 1):
            print(f"{idx}. {item.get('name')} ({item.get('ticker')}) - SID: {item.get('sid')}")
    
    def save_stock_data(self, stock_item: Dict[str, Any]) -> bool:
        if not stock_item.get('ticker'):
            return False
            
        stock_data = {
            'ticker': stock_item.get('ticker'),
            'sid': stock_item.get('sid'),
            'name': stock_item.get('name'),
            'sector': stock_item.get('sector'),
            'brands': stock_item.get('brands', []),
            'marketCap': stock_item.get('marketCap'),
            'quote': stock_item.get('quote', {}),
            'slug': stock_item.get('slug')
        }
        
        stock_folder = self.output_dir / stock_data['ticker'].lower()
        stock_folder.mkdir(exist_ok=True)
        
        # Save stock data with proper encoding
        with open(stock_folder / 'sData.json', 'w', encoding='utf-8') as f:
            json.dump(stock_data, f, indent=2, cls=JSONEncoder, ensure_ascii=False)

        # Fetch and save financial data
        fundamental = StockFundamentalData(str(self.output_dir))
        financial_success = fundamental.save_financial_data(stock_folder, stock_data['sid'])
        
        # Enrich stock data with company information
        if financial_success:
            try:
                print("Enriching stock data with company information...")
                self.data_enricher.enrich_stock_data(stock_data['ticker'])
                print("Stock data enrichment completed.")
            except Exception as e:
                print(f"Failed to enrich stock data: {e}")
        
        return financial_success
    
    def process_search(self, query: str) -> Optional[Dict[str, Any]]:
        items = self.search_stock(query)
        self.display_results(items)
        
        if not items:
            return None
            
        try:
            selection = int(input("\nSelect a stock number (or 0 to cancel): "))
            if 1 <= selection <= len(items):
                selected_stock = items[selection - 1]
                if self.save_stock_data(selected_stock):
                    return selected_stock
        except ValueError:
            print("Invalid selection")
        
        return None
