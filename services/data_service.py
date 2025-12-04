from pathlib import Path
import json
from typing import Dict, Optional
import requests
from core.config import API_ENDPOINTS, HTTP_HEADERS


class StockDataHandler:
    """Handle fetching and saving stock financial data"""
    
    def __init__(self, dir: str):
        self.dir = Path(dir)
        self.dir.mkdir(exist_ok=True)
        self.headers = HTTP_HEADERS

    def _save(self, data: Dict, path: Path) -> None:
        """Save data as JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: Path) -> Dict:
        """Load JSON data"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _fetch(self, sid: str, type: str, count: int = 15) -> Optional[Dict]:
        """Fetch data from API"""
        api_urls = {
            'quarterly': API_ENDPOINTS.QUARTERLY,
            'annual': API_ENDPOINTS.ANNUAL,
            'qt_growth': API_ENDPOINTS.QT_GROWTH,
            'an_growth': API_ENDPOINTS.AN_GROWTH,
            'balance_sheet': API_ENDPOINTS.BALANCE_SHEET,
            'bal_growth': API_ENDPOINTS.BAL_GROWTH,
            'cash_flow': API_ENDPOINTS.CASH_FLOW
        }
        
        url = api_urls.get(type, '').format(sid=sid)
        if not url:
            print(f"Invalid type: {type}")
            return None
        
        try:
            res = requests.get(url, headers=self.headers, params={"count": count}, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if not data.get('success'):
                print(f"API error: {data.get('message', 'Unknown')}")
                return None
            
            return data.get('data', [])
        except Exception as e:
            print(f"Fetch failed for {type}: {e}")
            return None

    def _fetch_summary(self, sid: str) -> Optional[Dict]:
        """Fetch summary data"""
        try:
            res = requests.get(API_ENDPOINTS.SUMMARY.format(sid=sid), headers=self.headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if not data.get('success'):
                print(f"API error: {data.get('message', 'Unknown')}")
                return None
            
            return data.get('data')
        except Exception as e:
            print(f"Summary fetch failed: {e}")
            return None

    def save_all(self, folder: Path, sid: str) -> bool:
        """Fetch and save all financial data"""
        types = {
            'quarterly': 'quarterly.json',
            'annual': 'annual.json',
            'qt_growth': 'qtGrowth.json',
            'an_growth': 'anGrowth.json',
            'balance_sheet': 'balancesheet.json',
            'bal_growth': 'balGrowth.json',
            'cash_flow': 'cashflow.json'
        }
        
        ok = True
        for type, file in types.items():
            data = self._fetch(sid, type)
            if data:
                self._save({f"{type}Data": data}, folder / file)
            else:
                ok = False
                print(f"Failed: {type} for {sid}")
        
        # Save summary
        summary = self._fetch_summary(sid)
        if summary:
            self._save(summary, folder / 'summary.json')
        
        # Update peers in sData.json
        if summary and summary.get('aboutAndPeers'):
            sdata_path = folder / 'sData.json'
            if sdata_path.exists():
                try:
                    sdata = self._load(sdata_path)
                    sdata['peers'] = [
                        {
                            'name': p.get('name'),
                            'sid': p.get('sid'),
                            'ticker': p.get('ticker'),
                            'slug': p.get('slug')
                        }
                        for p in summary['aboutAndPeers']
                    ]
                    self._save(sdata, sdata_path)
                except Exception as e:
                    print(f"Peers update failed: {e}")
                    ok = False
        
        return ok

