"""Utility Functions and Text/Data Sanitization Helpers.

Provides robust string sanitization, recursive nested data cleaning, custom JSON
encoding, and Excel dataframe listing search/persistence functions.

Google Python Style Guide Compliant.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Union
import pandas as pd
from core.config import FILE_PATHS

logger = logging.getLogger(__name__)

# Pre-compiled regex for control character removal to optimize performance
_CONTROL_CHAR_RE: re.Pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")


def sanitize_text(text: str) -> str:
    """Sanitizes raw text strings by replacing unencodable characters and stripping control chars.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned, UTF-8 compliant text string.
    """
    if not isinstance(text, str):
        return text

    # Handle UTF-8 encoding replacements
    clean_text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Remove non-printable control characters
    return _CONTROL_CHAR_RE.sub("", clean_text)


def sanitize_data(data: Any) -> Any:
    """Recursively sanitizes string values across nested dictionary and list structures.

    Args:
        data (Any): Arbitrary nested structure (dict, list, primitive).

    Returns:
        Any: Sanitized structural clone.
    """
    if isinstance(data, str):
        return sanitize_text(data)
    if isinstance(data, dict):
        return {sanitize_text(str(k)): sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling UTF-8 byte conversion for edge-case text payloads."""

    def default(self, obj: Any) -> Any:
        """Fallback encoder method.

        Args:
            obj (Any): Object to serialize.

        Returns:
            Any: Encoded representation or super delegation.
        """
        if isinstance(obj, str):
            return obj.encode("utf-8", errors="replace").decode("utf-8")
        return super().default(obj)


def find_stock_in_listings(ticker: str) -> Optional[Dict[str, Any]]:
    """Searches for a stock ticker within the filtered stock listings Excel database.

    Args:
        ticker (str): Stock ticker symbol or company name fragment.

    Returns:
        Optional[Dict[str, Any]]: Match dictionary if found, otherwise None.
    """
    if not ticker or not ticker.strip():
        logger.warning("Empty ticker string passed to find_stock_in_listings")
        return None

    target_file = FILE_PATHS.FILTERED_STOCK_LISTINGS
    if not os.path.exists(target_file):
        logger.error("Filtered stock listings database not found at path: %s", target_file)
        return None

    try:
        df = pd.read_excel(target_file)
        if "Stock Name" not in df.columns:
            logger.error("Excel file missing required 'Stock Name' column: %s", target_file)
            return None

        ticker_lower = ticker.strip().lower()
        matches = df[df["Stock Name"].astype(str).str.lower().str.contains(ticker_lower, na=False)]

        if not matches.empty:
            logger.info("Found %d matching stocks for ticker query '%s'", len(matches), ticker)
            return matches.iloc[0].to_dict()

        logger.info("No matching stock record found for query: '%s'", ticker)
        return None

    except Exception as exc:
        logger.error("Failed querying stock listings Excel database for '%s': %s", ticker, exc, exc_info=True)
        return None


def save_stock_to_listings(ticker: str, screener_url: str, stock_data: Dict[str, Any]) -> bool:
    """Appends a new stock metadata record to the filtered stock listings Excel file.

    Args:
        ticker (str): Stock ticker symbol.
        screener_url (str): Source URL or Screener reference link.
        stock_data (Dict[str, Any]): Key-value payload of additional metrics to persist.

    Returns:
        bool: True if save operation succeeded, False otherwise.
    """
    target_file = FILE_PATHS.FILTERED_STOCK_LISTINGS
    try:
        if os.path.exists(target_file):
            df = pd.read_excel(target_file)
        else:
            df = pd.DataFrame(columns=["Stock Name", "URL"])

        new_record: Dict[str, Any] = {
            "Stock Name": [ticker],
            "URL": [screener_url],
        }

        for col, value in stock_data.items():
            new_record[col] = [value]

        new_row_df = pd.DataFrame(new_record)
        updated_df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Ensure parent output directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_excel(target_file, index=False)
        logger.info("Successfully added stock ticker '%s' to listings database: %s", ticker, target_file)
        return True

    except Exception as exc:
        logger.error("Failed saving stock record '%s' to Excel listings: %s", ticker, exc, exc_info=True)
        return False
