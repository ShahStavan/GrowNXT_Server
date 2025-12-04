import requests
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
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
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return []
                
            # Return the complete items array without filtering
            return data.get('data', {}).get('items', [])
            
        except requests.RequestException as e:
            print(f"Request failed: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {str(e)}")
            return []

    def instant_search(self, query: str) -> None:
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
            
        ticker = stock_item.get('ticker')
        stock_folder = self.output_dir / ticker.lower()
        stock_folder.mkdir(exist_ok=True)
        
        sdata_file = stock_folder / 'sData.json'
        
        # Check if data already exists and is complete
        data_exists = self._check_data_exists(stock_folder)
        
        if data_exists:
            print(f"✓ Data already exists for {ticker}, skipping download and enrichment...")
            return True
        
        stock_data = {
            'ticker': ticker,
            'sid': stock_item.get('sid'),
            'name': stock_item.get('name'),
            'sector': stock_item.get('sector'),
            'brands': stock_item.get('brands', []),
            'marketCap': stock_item.get('marketCap'),
            'quote': stock_item.get('quote', {}),
            'slug': stock_item.get('slug')
        }
        
        # Save stock data 
        with open(sdata_file, 'w', encoding='utf-8') as f:
            json.dump(stock_data, f, indent=2, cls=JSONEncoder, ensure_ascii=False)

        # Fetch and save financial data
        fundamental = StockFundamentalData(str(self.output_dir))
        financial_success = fundamental.save_financial_data(stock_folder, stock_data['sid'])
        
        # Always attempt to enrich stock data, even if financial data fetch partially failed
        try:
            print(f"Enriching stock data with company information for {ticker}...")
            enrichment_success = self.data_enricher.enrich_stock_data(ticker)
            if enrichment_success:
                print(f"✓ Stock data enrichment completed for {ticker}")
            else:
                print(f"⚠ Stock data enrichment failed for {ticker}")
        except Exception as e:
            print(f"✗ Failed to enrich stock data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
        
        return financial_success
    
    def _check_data_exists(self, stock_folder: Path) -> bool:
        """Check if all required data files exist and sData.json has enrichment data"""
        required_files = [
            'sData.json',
            'quarterly.json',
            'annual.json',
            'balancesheet.json',
            'cashflow.json'
        ]
        
        # Check if all required files exist
        for file_name in required_files:
            if not (stock_folder / file_name).exists():
                return False
        
        # Check if sData.json has enrichment data
        try:
            sdata_file = stock_folder / 'sData.json'
            with open(sdata_file, 'r', encoding='utf-8') as f:
                sdata = json.load(f)
                # Check if enrichment data exists
                has_enrichment = (
                    'company_brief' in sdata and 
                    'strengths' in sdata and 
                    'weaknesses' in sdata and
                    sdata.get('company_brief')  # Check it's not empty
                )
                return has_enrichment
        except Exception:
            return False
    
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