from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from .base_scraper import BaseScraper
from concurrent.futures import ThreadPoolExecutor
import shutil

class PresentationScraper(BaseScraper):
    """Specialized class for scraping company presentations"""
    
    def __init__(self, stock_name: str, output_dir: Path, logger):
        super().__init__(logger)
        self.stock_name = stock_name
        self.output_dir = output_dir
        self.base_url = f"https://www.screener.in/company/{stock_name}/consolidated/"

    def get_presentations(self) -> List[Dict[str, Any]]:
        """Get all available presentations"""
        html_content = self._make_request(self.base_url)
        soup = self._parse_html(html_content)
        
        presentations = []
        concalls_section = soup.find('div', class_='documents concalls flex-column')
        
        if not concalls_section:
            return presentations

        for concall in concalls_section.find_all('li', class_='flex flex-gap-8 flex-wrap'):
            presentation = self._extract_presentation_info(concall)
            if presentation:
                presentations.append(presentation)

        return presentations

    def download_presentations(self, presentations: List[Dict]) -> bool:
        """Download presentations to presentations directory"""
        success = False
        try:
            for pres in presentations:
                url = pres['url']
                date_str = pres['date'].strftime('%Y_%m')
                filename = f"{self.stock_name}_presentation_{date_str}.pdf"
                filepath = self.presentations_dir / filename
                
                if not filepath.exists():
                    self.download_file(url, filepath)
                    success = True
                    
            return success
        except Exception as e:
            self.logger.error(f"Failed to download presentations: {e}")
            return False

    def _download_single_presentation(self, presentation: Dict[str, Any]) -> Dict[str, Any]:
        """Download a single presentation"""
        try:
            date_str = presentation['date'].strftime('%Y_%m')
            filename = f"date_ppt_{date_str}.pdf"  # Changed naming format
            filepath = self.presentations_dir / filename

            if not filepath.exists():
                self.download_file(presentation['url'], filepath)
                self.logger.info(f"Downloaded: {filename}")

            return {
                "date": presentation['date'].strftime('%Y-%m-%d'),
                "file": str(filepath)
            }
        except Exception as e:
            self.logger.error(f"Failed to download presentation: {e}")
            return None

    def _extract_presentation_info(self, concall_elem) -> Dict[str, Any]:
        """Extract presentation information from concall element"""
        try:
            date_div = concall_elem.find('div', class_='ink-600 font-size-15 font-weight-500 nowrap')
            if not date_div:
                return None

            date = datetime.strptime(date_div.text.strip(), '%b %Y')
            current_date = datetime.now()
            
            # Only process if it's from current year or last year
            if date.year < current_date.year - 1:
                return None
                
            # For presentations from current year, only take the latest month
            if date.year == current_date.year and date.month < current_date.month - 1:
                return None

            ppt_link = concall_elem.find('a', class_='concall-link', 
                                       href=lambda x: x and 'AnnPdfOpen.aspx' in x)
            if not ppt_link:
                return None

            return {
                'date': date,
                'url': self._format_link(ppt_link['href'])
            }
        except Exception as e:
            self.logger.error(f"Error extracting presentation info: {e}")
            return None

    def _format_link(self, href: str) -> str:
        """Format relative URLs to absolute URLs"""
        if href.startswith("http"):
            return href
        return f"https://www.screener.in{href}" if href.startswith("/") else None
