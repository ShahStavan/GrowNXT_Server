import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from utils import sanitize_text, sanitize_data, JSONEncoder

# Load environment variables from .env file
load_dotenv()

class CompanyDataEnricher:
    def __init__(self, data_dir: Path):
        """Initialize the company data enricher with data directory."""
        self.data_dir = data_dir
        api_key = os.getenv('GEMINI_API_KEY')
        model = os.getenv('MODEL_1')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model, 
            api_key=SecretStr(api_key)
        )

    def read_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Read existing stock data from sData.json"""
        stock_folder = self.data_dir / ticker.lower()
        file_path = stock_folder / 'sData.json'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Stock data not found for ticker: {ticker}")
            
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return json.load(f)
    
    def get_company_info(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Use LLM to get company information based on ticker and name."""
        # Sanitize inputs using utility function
        ticker = sanitize_text(ticker)
        company_name = sanitize_text(company_name)
        
        prompt = f"""
        Provide detailed information about {company_name} ({ticker}) in JSON format with the following structure:
        {{
            "brief_information": "Brief description of the company",
            "strengths": ["Strength 1", "Strength 2", ...],
            "weaknesses": ["Weakness 1", "Weakness 2", ...],
        }}
        
        Provide accurate and factual information only. If certain information is not available, leave it as null or empty array.
        Use UTF-8 encoding and avoid special characters that might cause JSON parsing errors.
        """
        
        response = self.llm.invoke(prompt)
        
        # Extract JSON content from the response
        try:
            # Find JSON content in the response text
            response_text = response.content
            
            # Find JSON between triple backticks if present
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_content = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_content = response_text
            
            # Parse JSON and sanitize data using utility function
            parsed_data = json.loads(json_content)
            return sanitize_data(parsed_data)
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response.content}")
            return {}
    
    def enrich_stock_data(self, ticker: str) -> bool:
        """Enrich existing stock data with company information."""
        try:
            # Get existing stock data
            stock_data = self.read_stock_data(ticker)
            company_name = stock_data.get('name', ticker)
            
            # Get additional company information
            company_info = self.get_company_info(ticker, company_name)
            
            # Update stock data with company information - only keeping essential fields
            stock_data.update({
                "company_brief": company_info.get("brief_information", ""),
                "strengths": company_info.get("strengths", []),
                "weaknesses": company_info.get("weaknesses", [])
            })
            
            # Sanitize entire stock data using utility function
            sanitized_data = sanitize_data(stock_data)
            
            # Save updated stock data
            stock_folder = self.data_dir / ticker.lower()
            file_path = stock_folder / 'sData.json'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, indent=2, cls=JSONEncoder, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error enriching stock data: {e}")
            return False
