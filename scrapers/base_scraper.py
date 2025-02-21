import requests
from bs4 import BeautifulSoup
from typing import Dict, Any
import logging
from pathlib import Path

class BaseScraper:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def _make_request(self, url: str) -> str:
        """Make HTTP request with error handling"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content"""
        return BeautifulSoup(html_content, 'html.parser')

    def _extract_table_data(self, soup: BeautifulSoup, section_id: str) -> list:
        """Generic method to extract table data"""
        section = soup.find('section', id=section_id)
        if not section:
            self.logger.warning(f"{section_id} section not found")
            return []

        table = section.find('table', class_='data-table')
        if not table:
            self.logger.warning(f"{section_id} table not found")
            return []

        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        data = []

        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            row_data = {headers[0]: cells[0].text.strip()}
            for header, cell in zip(headers[1:], cells[1:]):
                row_data[header] = cell.text.strip()
            data.append(row_data)

        return data

    def _format_link(self, href: str) -> str:
        """Format relative URLs to absolute URLs"""
        if not href:
            return ""
        if href.startswith("http"):
            return href
        return f"https://www.screener.in{href}" if href.startswith("/") else ""

    def download_file(self, url: str, output_path: Path) -> Path:
        """Download file with error handling"""
        try:
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                f.flush()
                
            import time
            time.sleep(1)  # Add small delay after file write
            return output_path
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            if output_path.exists():
                try:
                    output_path.unlink(missing_ok=True)
                except:
                    pass
            raise
