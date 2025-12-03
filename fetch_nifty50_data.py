from pathlib import Path
from data_handler import StockFundamentalData
from stock_search import StockSearch
from config import DATA_DIR
import time

# NIFTY 50 stock tickers
NIFTY_50_STOCKS = [
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

def fetch_all_nifty50_data():
    """Fetch and save financial data for all NIFTY 50 stocks"""
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Initialize stock search
    searcher = StockSearch(DATA_DIR)
    
    successful = []
    failed = []
    not_found = []
    
    print(f"Starting to fetch data for {len(NIFTY_50_STOCKS)} NIFTY 50 stocks...")
    print("=" * 70)
    
    for idx, ticker in enumerate(NIFTY_50_STOCKS, 1):
        print(f"\n[{idx}/{len(NIFTY_50_STOCKS)}] Processing {ticker}...")
        
        try:
            # Search for the stock to get the correct sid
            print(f"Searching for {ticker}...")
            search_results = searcher.search_stock(ticker)
            
            if not search_results:
                not_found.append(ticker)
                print(f"✗ Stock {ticker} not found in search results")
                time.sleep(0.5)
                continue
            
            # Find exact match or best match
            stock_match = None
            for result in search_results:
                if result.get('ticker') == ticker:
                    stock_match = result
                    break
            
            # If no exact match, use first result
            if not stock_match:
                stock_match = search_results[0]
                print(f"⚠ Using closest match: {stock_match.get('name')} ({stock_match.get('ticker')})")
            
            # Save stock data using the save_stock_data method
            success = searcher.save_stock_data(stock_match)
            
            if success:
                successful.append(ticker)
                print(f"✓ Successfully saved data for {ticker}")
            else:
                failed.append(ticker)
                print(f"✗ Failed to save complete data for {ticker}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            failed.append(ticker)
            print(f"✗ Error processing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("\nFETCH SUMMARY:")
    print(f"Total stocks: {len(NIFTY_50_STOCKS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Not Found: {len(not_found)}")
    
    if successful:
        print(f"\nSuccessful stocks: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed stocks: {', '.join(failed)}")
    
    if not_found:
        print(f"\nNot found stocks: {', '.join(not_found)}")
    
    print("\nData saved in:", DATA_DIR)

if __name__ == "__main__":
    fetch_all_nifty50_data()
