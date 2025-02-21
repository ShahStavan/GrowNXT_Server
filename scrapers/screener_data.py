from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class ScreenerDataScraper(BaseScraper):
    """Specialized class for scraping financial data from Screener"""
    
    def __init__(self, stock_name: str, logger):
        super().__init__(logger)
        self.stock_name = stock_name
        self.base_url = f"https://www.screener.in/company/{stock_name}/consolidated/"

    def get_financial_data(self) -> Dict[str, Any]:
        """Get all financial data for a stock"""
        html_content = self._make_request(self.base_url)
        soup = self._parse_html(html_content)

        return {
            "quarterly_results": self._extract_table_data(soup, 'quarters'),
            "profit_loss": self._extract_table_data(soup, 'profit-loss'),
            "balance_sheet": self._extract_table_data(soup, 'balance-sheet'),
            "cash_flow": self._extract_table_data(soup, 'cash-flow'),
            "ratios": self._extract_table_data(soup, 'ratios'),
            "shareholding": self._extract_shareholding_data(soup),
            "raw_pdf_links": self._extract_raw_pdf_links(soup)
        }

    def _extract_raw_pdf_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract raw PDF links from quarterly results"""
        raw_pdf_links = {}
        raw_pdf_row = soup.find('tr', class_='font-size-14 ink-600')
        
        if raw_pdf_row:
            cells = raw_pdf_row.find_all('td')[1:]
            try:
                headers = [th.text.strip() 
                          for th in soup.find('section', id='quarters').find('thead').find_all('th')[1:]]
                          
                for month, cell in zip(headers, cells):
                    link = cell.find('a')['href'] if cell.find('a') else ""
                    raw_pdf_links[month] = self._format_link(link)
            except Exception as e:
                self.logger.error(f"Error extracting PDF links: {e}")
                
        return raw_pdf_links

    def _extract_shareholding_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract shareholding pattern data"""
        shareholding_data = {'quarterly': {}, 'yearly': {}}
        section = soup.find('section', id='shareholding')
        
        if not section:
            return shareholding_data

        for period_type in ['quarterly', 'yearly']:
            div = section.find('div', id=f'{period_type}-shp')
            if div:
                table = div.find('table', class_='data-table')
                if table:
                    self._process_shareholding_table(table, shareholding_data[period_type])

        return shareholding_data

    def _process_shareholding_table(self, table: BeautifulSoup, data_dict: dict) -> None:
        """Process shareholding table data"""
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            category = cells[0].text.strip().replace('\xa0+', '').strip()
            
            if category == "No. of Shareholders":
                data_dict['shareholders'] = {
                    headers[i]: cells[i].text.strip()
                    for i in range(1, len(cells))
                }
            else:
                data_dict[category] = {
                    headers[i]: float(cells[i].text.strip().replace('%', ''))
                    for i in range(1, len(cells))
                }

    def download_latest_quarterly_pdf(self, output_dir: Path) -> Path:
        """Download latest quarterly PDF"""
        raw_data = self.get_financial_data()
        pdf_links = raw_data.get("raw_pdf_links", {})
        
        if not pdf_links:
            raise ValueError("No PDF links found")
            
        # Get latest date and URL
        latest_date = list(pdf_links.keys())[-1]
        latest_url = pdf_links[latest_date]
        
        if not latest_url:
            raise ValueError("No URL found for latest PDF")
            
        # Create filename with date
        formatted_date = latest_date.replace(' ', '_')
        pdf_path = output_dir / f"{self.stock_name}_{formatted_date}.pdf"
        
        # Download if doesn't exist
        if not pdf_path.exists():
            self.download_file(latest_url, pdf_path)
            self.logger.info(f"Downloaded quarterly PDF: {pdf_path.name}")
            
        return pdf_path
