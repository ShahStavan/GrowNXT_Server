from bs4 import BeautifulSoup, Tag
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .base_scraper import BaseScraper
import re

class ConcallScraper(BaseScraper):
    def __init__(self, stock_name: str, logger):
        super().__init__(logger)
        self.stock_name = stock_name
        self.base_url = f"https://www.screener.in/company/{stock_name}/consolidated/"

    def _parse_date(self, date_str: str) -> datetime:
        """Convert date string to datetime object"""
        try:
            return datetime.strptime(date_str.strip(), '%b %Y')
        except ValueError as e:
            self.logger.error(f"Failed to parse date: {date_str} - {e}")
            raise

    def _extract_links(self, item: Tag) -> Dict[str, str]:
        """Extract links for transcript and presentation only"""
        links = {}
        date_div = item.find('div', class_='ink-600 font-size-15 font-weight-500 nowrap')
        date_str = date_div.text.strip() if date_div else ""
        
        # Extract links with updated class names
        for link in item.find_all(['a'], class_='concall-link'):
            href = link.get('href', '')
            if href and 'AnnPdfOpen.aspx' in href:
                link_text = link.text.strip().lower()
                if 'transcript' in link_text:
                    links['transcript'] = href
                elif 'ppt' in link_text:
                    links['presentation'] = href
        
        return {
            'date': date_str,
            'links': links
        }

    def get_latest_concall_data(self) -> Optional[Dict]:
        """Get only the latest concall data"""
        html_content = self._make_request(self.base_url)
        soup = self._parse_html(html_content)
        
        # Update selector to match the HTML structure
        concalls_div = soup.find('div', class_='documents concalls flex-column')
        if not concalls_div:
            self.logger.warning("No concalls section found")
            return None
            
        # Look for list items within show-more-box
        show_more_box = concalls_div.find('div', class_='show-more-box')
        if not show_more_box:
            self.logger.warning("No show-more-box found")
            return None
            
        # Get the first (most recent) concall item
        latest_item = show_more_box.find('li', class_='flex flex-gap-8 flex-wrap')
        if not latest_item:
            self.logger.warning("No concall items found")
            return None
            
        try:
            data = self._extract_links(latest_item)
            if data['date']:
                data['date_obj'] = self._parse_date(data['date'])
                return data
        except Exception as e:
            self.logger.error(f"Failed to extract latest concall data: {e}")
            
        return None

    def download_document(self, url: str, filepath: Path) -> Optional[Path]:
        """Download document if it doesn't exist"""
        try:
            if not url:
                return None
                
            if not filepath.exists():
                return self.download_file(url, filepath)
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to download document: {e}")
            return None
