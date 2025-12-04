from api.search import StockSearch
from core.config import DATA_DIR
import time


NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDIGO", "INFY", "ITC", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TMPV", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN"
]


def _find_match(results, ticker):
    """Find exact or closest match"""
    for r in results:
        if r.get('ticker') == ticker:
            return r
    return results[0] if results else None


def fetch_all():
    """Fetch financial data for all NIFTY 50 stocks"""
    DATA_DIR.mkdir(exist_ok=True)
    searcher = StockSearch(DATA_DIR)
    
    ok, fail, missing = [], [], []
    total = len(NIFTY_50)
    
    print(f"Fetching {total} NIFTY 50 stocks...\n")
    
    for i, ticker in enumerate(NIFTY_50, 1):
        try:
            results = searcher.search_stock(ticker)
            
            if not results:
                missing.append(ticker)
                print(f"[{i}/{total}] ✗ {ticker} - Not found")
                time.sleep(0.5)
                continue
            
            match = _find_match(results, ticker)
            if not match:
                missing.append(ticker)
                print(f"[{i}/{total}] ✗ {ticker} - No match")
                continue
            
            # Save data
            if searcher.save_stock_data(match):
                ok.append(ticker)
                print(f"[{i}/{total}] ✓ {ticker}")
            else:
                fail.append(ticker)
                print(f"[{i}/{total}] ✗ {ticker} - Save failed")
            
            time.sleep(2)  # Rate limit
            
        except Exception as e:
            fail.append(ticker)
            print(f"[{i}/{total}] ✗ {ticker} - {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Total: {total} | OK: {len(ok)} | Failed: {len(fail)} | Missing: {len(missing)}")
    if fail:
        print(f"Failed: {', '.join(fail)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    fetch_all()
