import google.generativeai as genai
import os
import time
from typing import List, Any
import logging

DEFAULT_MODEL = "gemini-2.0-flash"

GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def initialize_genai(api_key: str):
    """Initialize the Gemini API with the provided key"""
    genai.configure(api_key=api_key)

def create_model(model_name: str = None) -> Any:
    """Create and return a model instance with proper configuration"""
    return genai.GenerativeModel(
        model_name=model_name or DEFAULT_MODEL,
        generation_config=GENERATION_CONFIG
    )

def wait_for_files_active(files: List[Any]) -> None:
    """Wait for uploaded files to be processed"""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("\nAll files ready")

def upload_and_wait_file(path: str, logger: logging.Logger) -> Any:
    """Upload file to Gemini and wait for processing"""
    try:
        file = genai.upload_file(path, mime_type="application/pdf")
        logger.info(f"Uploaded file: {file.display_name}")
        
        wait_for_files_active([file])
        return file
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise
