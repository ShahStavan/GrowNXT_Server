import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import PyPDF2
import chardet

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL_2")

def create_client():
    """Configure and return Google GenerativeAI"""
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    return genai

def create_content_part(file_content: str) -> str:
    """Create a content part from file content"""
    return file_content

def read_file_content(file_path: str) -> str:
    """Read file content based on file type"""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                return text
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.dumps(json.load(file))
        else:
            # For other files, detect encoding first
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""
