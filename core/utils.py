import json
import re
from typing import Any
import pandas as pd
import os
import logging
from core.config import FILE_PATHS

logger = logging.getLogger(__name__)

def sanitize_text(text: str) -> str:
    """Sanitize text to handle encoding issues."""
    if not isinstance(text, str):
        return text
    
    # Replace problematic characters
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    # Remove control characters
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

def sanitize_data(data: Any) -> Any:
    """Recursively sanitize all string values in nested data structures."""
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, dict):
        return {sanitize_text(k): sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    else:
        return data

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special characters and encoding issues."""
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        return super().default(obj)


def find_stock_in_listings(ticker):
    """Find if a stock exists in filtered_stock_listings and return its data."""
    try:
        if not os.path.exists(FILE_PATHS.FILTERED_STOCK_LISTINGS):
            logger.error(f"File not found: {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
            return None
        
        # Load the filtered stock listings
        df = pd.read_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS)
        
        # Search for exact or similar matches
        ticker_lower = ticker.lower()
        matching_stocks = df[df['Stock Name'].str.lower().str.contains(ticker_lower, na=False)]
        
        if len(matching_stocks) > 0:
            logger.info(f"Found {len(matching_stocks)} matching stocks for '{ticker}'")
            # Return the first match
            return matching_stocks.iloc[0].to_dict()
        else:
            logger.info(f"No matching stock found for '{ticker}'")
            return None
            
    except Exception as e:
        logger.error(f"Error finding stock in listings: {e}")
        return None

def save_stock_to_listings(ticker, screener_url, stock_data):
    """Save new stock data to filtered_stock_listings."""
    try:
        df = pd.read_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS)
        new_row = pd.DataFrame({
            'Stock Name': [ticker],
            'URL': [screener_url]
        })
        
        # Add the annual report and presentation data
        for col, value in stock_data.items():
            new_row[col] = value
        
        # Append to existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS, index=False)
        logger.info(f"Added new stock {ticker} to listings")
        return True
    except Exception as e:
        logger.error(f"Error saving stock to listings: {e}")
        return False
