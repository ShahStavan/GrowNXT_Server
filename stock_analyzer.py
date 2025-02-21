from pathlib import Path
import json
from datetime import datetime
import logging
from data_handler import StockDataHandler
from prompts import COMPANY_OVERVIEW_PROMPT, DETAILED_FINANCIAL_ANALYSIS_PROMPT, COMPANY_OVERVIEW_PROMPT_HINDI, DETAILED_FINANCIAL_ANALYSIS_PROMPT_HINDI
import google.generativeai as genai
from typing import Dict, Any, List
import time
import shutil
from llm_config import initialize_genai, create_model, wait_for_files_active, upload_and_wait_file
from scrapers.presentation_scraper import PresentationScraper
from scrapers.screener_data import ScreenerDataScraper
from scrapers.concall_scraper import ConcallScraper
from utils.financial_processor import FinancialDataProcessor
from utils.text_processor import TextProcessor
from utils.info_manager import InfoManager




class StockAnalyzer:
    def __init__(self, stock_name: str, output_dir: Path, api_key: str, logger: logging.Logger):
        self.stock_name = stock_name
        self.logger = logger
        
        # Create stock-specific folders and store paths
        self.stock_folder = output_dir / stock_name.lower()
        self.pdf_folder = self.stock_folder / "QuarterlyResultPdf"
        
        # Create all required directories
        self._create_directories()
        
        # Initialize components
        self.screener_scraper = ScreenerDataScraper(stock_name, logger)
        self.presentation_scraper = PresentationScraper(
            stock_name=stock_name, 
            output_dir=self.stock_folder, 
            logger=logger
        )
        self.data_handler = StockDataHandler(str(self.stock_folder))
        self.financial_processor = FinancialDataProcessor(logger)
        self.text_processor = TextProcessor()
        self.concall_scraper = ConcallScraper(stock_name, logger)
        self.info_manager = InfoManager(self.stock_folder, logger)
        
        # Initialize LLM
        initialize_genai(api_key)
        self.model = create_model()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.stock_folder,
            self.pdf_folder
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)

    def has_existing_analysis(self) -> bool:
        """Check if analysis results already exist"""
        analysis_file = self.stock_folder / 'stock_data.json'
        return analysis_file.exists()

    def has_existing_presentations(self) -> bool:
        """Check if latest presentation exists"""
        return any(self.stock_folder.glob("Latest_Info/latest_presentation_*.pdf"))

    def download_latest_report(self) -> Path:
        """Download latest quarterly report"""
        return self.screener_scraper.download_latest_quarterly_pdf(self.pdf_folder)

    def analyze_pdf(self, pdf_path: Path, company_name: str) -> Dict[str, Any]:
        """Analyze PDF using Gemini chat session"""
        try:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            self.logger.info(f"Starting analysis of {pdf_path}")
            file = upload_and_wait_file(str(pdf_path), self.logger)
            
            if not file:
                raise ValueError("File upload failed")
            
            # Get English analysis
            english_data = self._get_language_analysis(file, 
                COMPANY_OVERVIEW_PROMPT, 
                DETAILED_FINANCIAL_ANALYSIS_PROMPT
            )
            
            # Get Hindi analysis
            hindi_data = self._get_language_analysis(file, 
                COMPANY_OVERVIEW_PROMPT_HINDI, 
                DETAILED_FINANCIAL_ANALYSIS_PROMPT_HINDI
            )
            
            # Format bilingual response
            analysis = {
                "stock_name": company_name,
                "analysis_date": datetime.now().isoformat(),
                "source": str(pdf_path),
                "english": {
                    "overview_points": english_data["overview"],
                    "business_model": english_data["business_model"],
                    "financial_performance": english_data["financial_performance"],
                    "operational_capabilities": english_data["operational_capabilities"],
                    "sectorial_analysis": english_data["sectorial_analysis"]
                },
                "hindi": {
                    "overview_points": hindi_data["overview"],
                    "business_model": hindi_data["business_model"],
                    "financial_performance": hindi_data["financial_performance"],
                    "operational_capabilities": hindi_data["operational_capabilities"],
                    "sectorial_analysis": hindi_data["sectorial_analysis"]
                }
            }
            
            # Validate response has content
            if not self._validate_bilingual_content(analysis):
                raise ValueError("No analysis points extracted")
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to analyze presentation: {str(e)}")

    def _extract_clean_points(self, data: Any) -> List[str]:
        """Clean and extract only the main points without metadata"""
        points = []
        
        try:
            if isinstance(data, str):
                # Try to parse if it's a JSON string
                try:
                    data = json.loads(data)
                except:
                    # Clean and add if it's a plain string
                    return [self._clean_point(data)]

            if isinstance(data, dict):
                # If it's a point/analysis structure, take only the point
                if 'Point' in data:
                    points.append(self._clean_point(data['Point']))
                else:
                    # For other dicts, process all values
                    for value in data.values():
                        points.extend(self._extract_clean_points(value))
            
            elif isinstance(data, list):
                for item in data:
                    points.extend(self._extract_clean_points(item))

        except Exception as e:
            self.logger.error(f"Error extracting points: {e}")
            return []

        return [p for p in points if p]

    def _clean_point(self, text: str) -> str:
        """Clean and format a single point"""
        # Remove JSON formatting and quotes
        clean = (text.replace("{", "")
                    .replace("}", "")
                    .replace('"', '')
                    .replace("'", "")
                    .strip())
        
        # Remove metadata prefixes
        prefixes_to_remove = ['Point:', 'Analysis:', 'Source:', 'Details:']
        for prefix in prefixes_to_remove:
            if clean.lower().startswith(prefix.lower()):
                clean = clean[len(prefix):].strip()
        
        # Remove any remaining JSON artifacts
        if ': ' in clean:
            parts = clean.split(': ')
            if len(parts) > 1:
                clean = parts[1]
        
        return clean.strip()

    def _format_section_data(self, section_data: Any) -> List[str]:
        """Format section data into clean bullet points"""
        # First extract all points
        raw_points = self._extract_clean_points(section_data)
        
        # Clean and format points
        formatted_points = []
        for point in raw_points:
            # Remove any remaining JSON artifacts and clean up formatting
            clean_point = (point.replace("{", "")
                               .replace("}", "")
                               .replace('"', '')
                               .replace("'", "")
                               .strip())
                               
            if ':' in clean_point:
                # If it's a key-value pair, just take the value
                clean_point = clean_point.split(":", 1)[1].strip()
                
            if clean_point and not any(clean_point.lower().startswith(x) for x in ['source', 'page', 'details']):
                formatted_points.append(clean_point)
        
        return formatted_points

    def _get_language_analysis(self, file: Any, overview_prompt: str, 
                             financial_prompt: str) -> Dict[str, Any]:
        """Get analysis in specified language"""
        try:
            overview_data = self._get_overview_analysis(file, overview_prompt)
            financial_data = self._get_financial_analysis(file, financial_prompt)
            
            # Return formatted data with proper bullet points
            analysis = {
                "overview": self._format_section_data(overview_data.get("shortDescription", [])),
                "business_model": self._format_section_data(
                    financial_data.get("Business Model & Strategy", {})
                ),
                "financial_performance": self._format_section_data(
                    financial_data.get("Financial Performance & Projections", {})
                ),
                "operational_capabilities": self._format_section_data(
                    financial_data.get("Operational Capabilities & Risks", {})
                ),
                "sectorial_analysis": self._format_section_data(
                    financial_data.get("Sectorial Analysis", {})
                )
            }
            
            # Validate and clean empty sections
            return {k: v for k, v in analysis.items() if v}
                
        except Exception as e:
            self.logger.error(f"Language analysis failed: {e}")
            raise

    def _get_overview_analysis(self, file: Any, prompt: str) -> Dict[str, Any]:
        """Get overview analysis"""
        overview_chat = self.model.start_chat(history=[{"role": "user", "parts": [file, prompt]}])
        overview_response = overview_chat.send_message(
            "Return the analysis in JSON format with shortDescription field containing bullet points."
        )
        return self.text_processor.extract_json_from_text(
            overview_response.text, 
            is_financial=False, 
            logger=self.logger
        )

    def _get_financial_analysis(self, file: Any, prompt: str) -> Dict[str, Any]:
        """Get financial analysis"""
        financial_chat = self.model.start_chat(history=[{"role": "user", "parts": [file, prompt]}])
        financial_response = financial_chat.send_message(
            "Return the detailed financial analysis in the specified JSON format."
        )
        return self.text_processor.extract_json_from_text(
            financial_response.text, 
            is_financial=True,
            logger=self.logger
        )

    def _validate_bilingual_content(self, analysis: Dict[str, Any]) -> bool:
        """Validate both English and Hindi content exists"""
        required_fields = [
            "overview_points",
            "business_model",
            "financial_performance",
            "operational_capabilities",
            "sectorial_analysis"
        ]
        
        return all(
            lang in analysis and
            isinstance(analysis[lang], dict) and
            all(field in analysis[lang] and len(analysis[lang][field]) > 0 
                for field in required_fields)
            for lang in ['english', 'hindi']
        )

    def save_results(self, results: dict) -> Path:
        """Save analysis results"""
        return self.data_handler.process_data(self.stock_name, results)

    def get_latest_presentation(self) -> Path:
        """Get the most recent presentation from Latest_Info"""
        # Look for date_ppt files instead of latest_presentation
        latest_files = list(self.stock_folder.glob("Latest_Info/date_ppt_*.pdf"))
        
        if not latest_files:
            # If no presentation in Latest_Info, download latest
            self.update_latest_info()
            latest_files = list(self.stock_folder.glob("Latest_Info/date_ppt_*.pdf"))
            
        if latest_files:
            # Sort by date in filename and get most recent
            latest = max(latest_files, 
                        key=lambda x: datetime.strptime(x.stem.split('_')[-2] + '_' + x.stem.split('_')[-1], '%Y_%m'))
            return latest
            
        raise ValueError("No presentation found after update")

    def extract_financial_data(self) -> Dict[str, Any]:
        """Extract and process financial data"""
        try:
            # Get raw data
            raw_data = self.screener_scraper.get_financial_data()
            
            # Process data
            processed_data = self.financial_processor.process_data(raw_data)
            
            # Save processed data
            self.financial_processor.save_financial_data(processed_data, self.stock_folder)
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Failed to extract financial data: {e}")
            raise

    def has_complete_analysis(self) -> bool:
        """Check if complete analysis data exists"""
        try:
            analysis_file = self.stock_folder / 'stock_data.json'
            if not analysis_file.exists():
                return False
                
            with open(analysis_file) as f:
                data = json.load(f)
                
            # Check if all required fields exist and have content
            required_fields = [
                "overview_points",
                "business_model",
                "financial_performance",
                "operational_capabilities",
                "sectorial_analysis"
            ]
            
            return all(
                field in data and 
                isinstance(data[field], list) and 
                len(data[field]) > 0 
                for field in required_fields
            )
        except Exception as e:
            self.logger.error(f"Error checking analysis completeness: {e}")
            return False

    def update_latest_info(self) -> None:
        """Update Latest_Info with most recent presentation and transcript"""
        try:
            # Get latest concall data
            latest_concall = self.concall_scraper.get_latest_concall_data()
            if not latest_concall:
                self.logger.warning("No concall data found")
                return

            date_str = latest_concall['date_obj'].strftime('%Y_%m')
            links = latest_concall['links']
            
            file_types = {
                'presentation': 'date_ppt',
                'transcript': 'date_transcript'
            }
            
            for doc_type, prefix in file_types.items():
                url = links.get(doc_type)
                if url:
                    # Download document
                    temp_path = self.stock_folder / f"temp_{doc_type}.pdf"
                    downloaded = self.concall_scraper.download_document(url, temp_path)
                    
                    if downloaded:
                        try:
                            # Create final path and ensure directory exists
                            final_name = f"{prefix}_{date_str}.pdf"
                            final_path = self.stock_folder / "Latest_Info" / final_name
                            final_path.parent.mkdir(exist_ok=True)
                            
                            # Use copy2 instead of move, then remove original
                            if temp_path.exists():
                                import time
                                time.sleep(1)  # Add small delay to ensure file is released
                                shutil.copy2(temp_path, final_path)
                                time.sleep(1)  # Add small delay before deletion
                                temp_path.unlink(missing_ok=True)
                                self.logger.info(f"Saved {doc_type} as {final_name}")
                        except Exception as e:
                            self.logger.error(f"Error processing {doc_type}: {e}")
                            if temp_path.exists():
                                try:
                                    temp_path.unlink(missing_ok=True)
                                except:
                                    pass
                        
            self.logger.info(f"Latest info updated successfully for {date_str}")
            
        except Exception as e:
            self.logger.error(f"Failed to update latest info: {e}")
            raise

    def run_analysis(self) -> dict:
        """Run the complete analysis workflow"""
        try:
            # Update latest info first
            self.update_latest_info()
            
            # Get latest presentation for analysis
            presentation_path = self.get_latest_presentation()
            self.logger.info(f"Using latest presentation: {presentation_path.name}")
            
            # Analyze through Gemini
            self.logger.info("Starting Gemini analysis of presentation")
            results = self.analyze_pdf(presentation_path, self.stock_name)
            
            # Save results
            output_path = self.save_results(results)
            self.logger.info(f"Results saved to: {output_path}")

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
