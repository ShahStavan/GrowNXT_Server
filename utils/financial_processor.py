from typing import Dict, Any, List, Union
from pathlib import Path
import json
from .text_processor import TextProcessor
import logging

class FinancialDataProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.text_processor = TextProcessor()

    def _process_shareholding_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process shareholding data"""
        processed = {'quarterly': {}, 'yearly': {}}
        
        for period_type in ['quarterly', 'yearly']:
            for category, values in data[period_type].items():
                if category == 'shareholders':
                    processed[period_type][category] = {
                        date: self.text_processor.extract_numeric_value(value)
                        for date, value in values.items()
                    }
                else:
                    # Shareholding values are already floats, no need to process
                    processed[period_type][category] = values
                    
        return processed

    def _safe_process_value(self, value: Any) -> Union[float, str, Any]:
        """Safely process a value, handling various types"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return self.text_processor.process_table_cell(value)
        return value

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all financial data"""
        processed = {}
        
        for key, value in data.items():
            if not value:  # Skip empty data
                continue
                
            if key in ['quarterly_results', 'profit_loss', 'balance_sheet', 
                      'cash_flow', 'ratios']:
                processed[key] = [
                    {k: self._safe_process_value(v) for k, v in row.items()}
                    for row in value
                ]
            elif key == 'shareholding':
                processed[key] = self._process_shareholding_data(value)
            else:
                processed[key] = value
                
        return processed

    def save_financial_data(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Save processed financial data to separate files"""
        for key, value in data.items():
            if not value:  # Skip empty data
                continue
                
            filepath = output_dir / f"{key}.json"
            with open(filepath, 'w') as f:
                json.dump(value, f, indent=2)
            self.logger.info(f"Saved {key} data to {filepath}")
