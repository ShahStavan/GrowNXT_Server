from pathlib import Path
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseDataHandler(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save JSON data with proper UTF-8 encoding"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data with proper UTF-8 encoding"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @abstractmethod
    def process_data(self):
        pass

class StockDataHandler(BaseDataHandler):
    def save_stock_data(self, data: Dict[str, Any]) -> Path:
        # Save directly to stock folder (output_dir is already stock-specific)
        filepath = self.output_dir / 'stock_data.json'
        self.save_json(data, filepath)
        return filepath

    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load existing JSON data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load existing analysis: {e}")

    def process_data(self, company_name: str, data: Dict[str, Any]) -> Path:
        return self.save_stock_data(data)
