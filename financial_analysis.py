from pathlib import Path
import logging
import google.generativeai as genai
from llm_config import create_client, create_content_part, read_file_content, DEFAULT_MODEL
from prompts import ANALYSIS_PROMPT, DCF_PROMPT

def generate_financial_analysis(stock_folder: Path, mapping_file: Path) -> str:
    """Generate financial analysis for a given stock and save as report.md"""
    try:
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        
        # List of files to process
        file_types = [
            'qtGrowth.json', 'balGrowth.json', 'balancesheet.json',
            'annual.json', 'anGrowth.json', 'quarterly.json', 
            'sData.json', 'cashflow.json'
        ]
        
        # Read all files content
        content_parts = []
        for file_name in file_types:
            file_path = stock_folder / file_name
            if file_path.exists():
                content = read_file_content(str(file_path))
                content_parts.append(create_content_part(content))
        
        # Add PDF content if exists
        pdf_files = list(stock_folder.glob('*.pdf'))
        if pdf_files:
            content = read_file_content(str(pdf_files[0]))
            content_parts.append(create_content_part(content))
            
        # Add mapping file content
        mapping_content = read_file_content(str(mapping_file))
        content_parts.append(create_content_part(mapping_content))
        
        # Add the analysis prompt
        content_parts.append(ANALYSIS_PROMPT)
        
        # Generate content
        prompt = "\n".join(content_parts)
        response = model.generate_content(prompt)
            
        # Save the response as report.md
        report_file = stock_folder / 'report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
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
