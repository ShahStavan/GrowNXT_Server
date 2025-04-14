import threading
import pandas as pd
import os
import requests
from pathlib import Path
from config import FILE_PATHS, DATA_DIR
from ticker_scraper import process_single_stock
import logging
from utils import find_stock_in_listings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("background_tasks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("background_tasks")

def download_file(url, save_path):
    """Download a file from url and save it to the specified path."""
    if not url or not url.strip():
        logger.error("Invalid URL provided")
        return False

    try:
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add user-agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the file
        logger.info(f"Downloading file from {url}")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"File saved to {save_path}")
        return True
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error occurred: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {e}")
    
    return False


def check_existing_downloads(ticker_dir):
    """Check if documents are already downloaded for a given ticker directory."""
    try:
        # Check if both report and presentation directories exist and contain files
        report_dir = ticker_dir / "report"
        ppt_dir = ticker_dir / "ppt"
        
        if not (report_dir.exists() and ppt_dir.exists()):
            return False
            
        # Check if there are any PDF files in either directory
        report_files = list(report_dir.glob("*.pdf"))
        ppt_files = list(ppt_dir.glob("*.pdf"))
        
        return len(report_files) > 0 or len(ppt_files) > 0
        
    except Exception as e:
        logger.error(f"Error checking existing downloads: {e}")
        return False


def download_stock_documents(ticker, stock_data=None):
    """Download the latest report and presentation for a stock in background."""
    thread = threading.Thread(
        target=_download_stock_documents_task,
        args=(ticker, stock_data),
        daemon=True
    )
    thread.start()
    return True

def _download_stock_documents_task(ticker, stock_data=None):
    """Background task to download stock documents."""
    try:
        logger.info(f"Starting background download task for {ticker}")
        
        # First check if stock exists in the listings or use provided stock data
        stock_info = stock_data if stock_data else find_stock_in_listings(ticker)
        
        if not stock_info:
            logger.info(f"Stock '{ticker}' not found in listings, attempting to scrape...")
            # Use process_single_stock to scrape and save the data
            process_single_stock(ticker)
            # Check if we found the stock after scraping
            stock_info = find_stock_in_listings(ticker)
            
            if not stock_info:
                logger.error(f"Failed to find or scrape stock information for '{ticker}'")
                return False
        
        # Create directories for saving documents
        ticker_dir = DATA_DIR / ticker.lower()
        report_dir = ticker_dir / "report"
        ppt_dir = ticker_dir / "ppt"
        
        report_dir.mkdir(parents=True, exist_ok=True)
        ppt_dir.mkdir(parents=True, exist_ok=True)
        
        # Download latest report if available
        latest_report_url = stock_info.get('Latest Report')
        if latest_report_url and isinstance(latest_report_url, str) and latest_report_url.strip():
            report_filename = f"LatestReport_{ticker}.pdf"
            report_path = report_dir / report_filename
            
            logger.info(f"Downloading latest report for {ticker}")
            download_success = download_file(latest_report_url, report_path)
            if download_success:
                logger.info(f"Successfully downloaded latest report for {ticker}")
            else:
                logger.error(f"Failed to download latest report for {ticker}")
        else:
            logger.info(f"No latest report URL available for {ticker}")
        
        # Download latest presentation if available
        latest_pres_url = stock_info.get('Latest Presentation')
        if latest_pres_url and isinstance(latest_pres_url, str) and latest_pres_url.strip():
            pres_filename = f"LatestPresentation_{ticker}.pdf"
            pres_path = ppt_dir / pres_filename
            
            logger.info(f"Downloading latest presentation for {ticker}")
            download_success = download_file(latest_pres_url, pres_path)
            if download_success:
                logger.info(f"Successfully downloaded latest presentation for {ticker}")
            else:
                logger.error(f"Failed to download latest presentation for {ticker}")
        else:
            logger.info(f"No latest presentation URL available for {ticker}")
        
        logger.info(f"Completed background download task for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error in background download task for {ticker}: {e}")
        return False
