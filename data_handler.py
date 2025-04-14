from pathlib import Path
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from config import API_ENDPOINTS, HTTP_HEADERS

class BaseDataHandler(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save JSON data with proper UTF-8 encoding"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data with proper UTF-8 encoding"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @abstractmethod
    def process_data(self):
        pass

class StockFundamentalData(BaseDataHandler):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.headers = HTTP_HEADERS

    def fetch_financial_data(self, sid: str, data_type: str = 'quarterly', count: int = 15) -> Optional[Dict[str, Any]]:
        """Fetch financial data based on type (quarterly, annual, qt_growth, an_growth, balance_sheet, bal_growth, or cash_flow)"""
        try:
            api_map = {
                'quarterly': API_ENDPOINTS.QUARTERLY,
                'annual': API_ENDPOINTS.ANNUAL,
                'qt_growth': API_ENDPOINTS.QT_GROWTH,
                'an_growth': API_ENDPOINTS.AN_GROWTH,
                'balance_sheet': API_ENDPOINTS.BALANCE_SHEET,
                'bal_growth': API_ENDPOINTS.BAL_GROWTH,
                'cash_flow': API_ENDPOINTS.CASH_FLOW
            }
            
            url = api_map.get(data_type, '').format(sid=sid)
            if not url:
                print(f"Invalid data type: {data_type}")
                return None
                
            params = {"count": count}
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
                
            return data.get('data', [])
            
        except Exception as e:
            print(f"Failed to fetch {data_type} data: {e}")
            return None

    def fetch_summary_data(self, sid: str) -> Optional[Dict[str, Any]]:
        """Fetch summary data for the stock"""
        try:
            response = requests.get(
                API_ENDPOINTS.SUMMARY.format(sid=sid),
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
                
            return data.get('data')
            
        except Exception as e:
            print(f"Failed to fetch summary data: {e}")
            return None

    def save_financial_data(self, stock_folder: Path, sid: str) -> bool:
        """Save all financial data types including summary"""
        data_types = {
            'quarterly': 'quarterly.json',
            'annual': 'annual.json',
            'qt_growth': 'qtGrowth.json',
            'an_growth': 'anGrowth.json',
            'balance_sheet': 'balancesheet.json',
            'bal_growth': 'balGrowth.json',
            'cash_flow': 'cashflow.json'
        }
        
        success = True
        for data_type, filename in data_types.items():
            data = self.fetch_financial_data(sid, data_type=data_type)
            if data:
                self.save_json({f"{data_type}Data": data}, stock_folder / filename)
            else:
                success = False
                print(f"Failed to fetch {data_type} data for {sid}")
        
        # Add summary data fetch and save
        summary_data = self.fetch_summary_data(sid)
        if summary_data:
            self.save_json(summary_data, stock_folder / 'summary.json')
        
        # Update sData.json with peers info if summary exists
        if summary_data and summary_data.get('aboutAndPeers'):
            sdata_path = stock_folder / 'sData.json'
            if sdata_path.exists():
                try:
                    sdata = self.load_json(sdata_path)
                    peers = []
                    for peer in summary_data['aboutAndPeers']:
                        peers.append({
                            'name': peer.get('name'),
                            'sid': peer.get('sid'),
                            'ticker': peer.get('ticker'),
                            'slug': peer.get('slug')
                        })
                    sdata['peers'] = peers
                    self.save_json(sdata, sdata_path)
                except Exception as e:
                    print(f"Failed to update peers in sData.json: {e}")
                    success = False
        
        return success

    def process_data(self, sid: str = None, data: Dict[str, Any] = None) -> bool:
        if sid and isinstance(sid, str):
            return self.save_financial_data(self.output_dir, sid)
        return False

class StockDataHandler(BaseDataHandler):
    def save_stock_data(self, data: Dict[str, Any]) -> Path:
        # Save directly to stock folder (output_dir is already stock-specific)
        filepath = self.output_dir / 'stock_data.json'
        self.save_json(data, filepath)
        return filepath

    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load existing JSON data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load existing analysis: {e}")

    def process_data(self, company_name: str, data: Dict[str, Any]) -> Path:
        folder_path = self.save_stock_data(data)
        if 'sid' in data:
            fundamental = StockFundamentalData(str(self.output_dir))
            fundamental.save_financial_data(folder_path.parent, data['sid'])
        return folder_path
