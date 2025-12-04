import json
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from core.utils import sanitize_text, sanitize_data, JSONEncoder

load_dotenv()


class CompanyEnricher:
    """Enrich stock data with company info using LLM"""
    
    def __init__(self, dir: Path):
        self.dir = dir
        key = os.getenv('GEMINI_API_KEY')
        model = os.getenv('MODEL_1')
        if not key:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(key))

    def _read(self, ticker: str) -> Dict:
        """Read sData.json"""
        path = self.dir / ticker.lower() / 'sData.json'
        if not path.exists():
            raise FileNotFoundError(f"No data for {ticker}")
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return json.load(f)
    
    def _fetch_info(self, ticker: str, name: str) -> Dict:
        """Get company info from LLM"""
        ticker = sanitize_text(ticker)
        name = sanitize_text(name)
        
        prompt = f"""
        Provide info about {name} ({ticker}) in JSON:
        {{
            "brief_information": "Company description",
            "strengths": ["Strength 1", "Strength 2", ...],
            "weaknesses": ["Weakness 1", "Weakness 2", ...]
        }}
        Provide factual info only. Use UTF-8 encoding.
        """
        
        res = self.llm.invoke(prompt)
        
        try:
            txt = res.content
            
            # Extract JSON from markdown blocks
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0].strip()
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0].strip()
            
            data = json.loads(txt)
            return sanitize_data(data)
        except Exception as e:
            print(f"Parse error: {e}")
            return {}
    
    def enrich(self, ticker: str) -> bool:
        """Add company info to stock data"""
        try:
            data = self._read(ticker)
            name = data.get('name', ticker)
            
            # Fetch and update
            info = self._fetch_info(ticker, name)
            data.update({
                "company_brief": info.get("brief_information", ""),
                "strengths": info.get("strengths", []),
                "weaknesses": info.get("weaknesses", [])
            })
            
            # Save
            data = sanitize_data(data)
            path = self.dir / ticker.lower() / 'sData.json'
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, cls=JSONEncoder, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Enrich failed: {e}")
            return False
