import os
import google.generativeai as genai
from dotenv import load_dotenv

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
    """Read and return file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
