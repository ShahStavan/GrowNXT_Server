from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(r"D:/Stock_Fundamental")
OUTPUT_FOLDER = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
