from pathlib import Path
import logging
import time
from llm_config import create_client, create_content_part, read_file_content, DEFAULT_MODEL
from prompts import ANALYSIS_PROMPT, DCF_PROMPT

logging.basicConfig(level=logging.INFO)

def generate_financial_analysis(stock_folder: Path, mapping_file: Path) -> str:
    """Generate financial analysis for a given stock and save as report.md"""
    try:
        logging.info(f"Starting financial analysis for folder: {stock_folder}")
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        
        # List of files to process
        json_files = [
            'qtGrowth.json', 'balGrowth.json', 'balancesheet.json',
            'annual.json', 'anGrowth.json', 'quarterly.json', 
            'sData.json', 'cashflow.json', 'summary.json',
        ]
        
        pdf_files = ['annual_report.pdf', 'presentation.pdf']
        
        # Read all files content
        content_parts = []
        
        # Process JSON files
        for file_name in json_files:
            file_path = stock_folder / file_name
            if file_path.exists():
                content = read_file_content(str(file_path))
                content_parts.append(create_content_part(content))
        
        # Enhanced PDF processing
        pdf_files = ['annual_report.pdf', 'presentation.pdf']
        pdf_contents = []
        
        for pdf_file in pdf_files:
            file_path = stock_folder / pdf_file
            if file_path.exists():
                # Add retry mechanism for newly downloaded files
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        logging.info(f"Processing PDF file (attempt {attempt + 1}): {file_path}")
                        content = read_file_content(str(file_path))
                        if content and len(content.strip()) > 0:
                            logging.info(f"Successfully extracted content from {pdf_file} (Size: {len(content)} chars)")
                            pdf_contents.append(create_content_part(
                                f"Analysis from {pdf_file}:\n{content}"
                            ))
                            break
                        else:
                            logging.warning(f"Empty content from {pdf_file}, retrying...")
                            time.sleep(1)  # Wait a bit before retrying
                    except Exception as pdf_error:
                        logging.error(f"Error processing {pdf_file} (attempt {attempt + 1}): {pdf_error}")
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to process {pdf_file} after {max_retries} attempts")
        
        # Add PDF contents to the full content list
        content_parts.extend(pdf_contents)
        
        # Add mapping file content
        mapping_content = read_file_content(str(mapping_file))
        content_parts.append(create_content_part(mapping_content))
        
        # Add the analysis prompt
        content_parts.append(ANALYSIS_PROMPT)
        
        # Generate content
        prompt = "\n".join(content_parts)
        response = model.generate_content(prompt)
            
        # Save the response as report.md with error handling
        try:
            report_content = response.text
            if isinstance(report_content, str):
                report_file = stock_folder / 'report.md'
                with open(report_file, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(report_content)
            else:
                logging.warning(f"Unexpected response type: {type(report_content)}")
                
            return report_content
            
        except Exception as write_error:
            logging.error(f"Error writing report: {write_error}")
            # Return the content even if writing fails
            return response.text
        
    except Exception as e:
        logging.error(f"Failed to generate analysis: {e}")
        raise

def generate_dcf_analysis(stock_folder: Path) -> str:
    """Generate DCF analysis for a given stock and save as dcf_report.md"""
    try:
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        
        # Required files list
        required_files = [
            'annual.json',
            'sData.json',
            'cashflow.json',
            'balancesheet.json'
        ]
        
        content_parts = []
        missing_files = []
        
        # Add mapping file content
        mapping_file = Path(__file__).parent / 'mapping.json'
        if mapping_file.exists():
            mapping_content = read_file_content(str(mapping_file))
            content_parts.append(create_content_part(mapping_content))
        
        # Add required files content
        for file_name in required_files:
            file_path = stock_folder / file_name
            if file_path.exists():
                content = read_file_content(str(file_path))
                content_parts.append(create_content_part(content))
            else:
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Required files missing for DCF analysis: {', '.join(missing_files)}")
        
        # Add DCF prompt
        content_parts.append(DCF_PROMPT)
        
        # Generate content
        prompt = "\n".join(content_parts)
        response = model.generate_content(prompt)
            
        # Save the response as dcf_report.md
        report_file = stock_folder / 'dcf_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        return response.text
        
    except Exception as e:
        logging.error(f"Failed to generate DCF analysis: {e}")
        raise
