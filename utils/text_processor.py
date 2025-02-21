from typing import Any, Dict, Union, List
import re
import json
import logging

class TextProcessor:
    @staticmethod
    def clean_non_breaking_space(text: str) -> str:
        """Remove non-breaking space characters"""
        if isinstance(text, (int, float)):
            return text
        return text.replace('\xa0', ' ').strip()

    @staticmethod
    def clean_currency_symbols(text: Union[str, int, float]) -> Union[str, int, float]:
        """Clean currency symbols and normalize text"""
        if isinstance(text, (int, float)):
            return text
        return text.replace('₹', 'Rs.').replace('\u20b9', 'Rs.')

    @staticmethod
    def extract_numeric_value(text: Union[str, int, float]) -> Union[float, str]:
        """Extract numeric value from text"""
        if isinstance(text, (int, float)):
            return float(text)
            
        # Remove commas and other formatting
        cleaned = re.sub(r'[^\d.-]', '', str(text))
        try:
            return float(cleaned) if cleaned else text
        except ValueError:
            return text

    @staticmethod
    def process_table_cell(cell_text: Union[str, int, float]) -> Union[float, str]:
        """Process individual table cell content"""
        if isinstance(cell_text, (int, float)):
            return float(cell_text)
            
        text = TextProcessor.clean_non_breaking_space(str(cell_text))
        text = TextProcessor.clean_currency_symbols(text)
        
        # If text contains %, convert to float
        if isinstance(text, str) and '%' in text:
            text = text.replace('%', '')
            try:
                return float(text)
            except ValueError:
                return text
                
        return TextProcessor.extract_numeric_value(text)

    @staticmethod
    def process_financial_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process financial data tables"""
        processed_data = []
        for row in data:
            processed_row = {}
            for key, value in row.items():
                # Process the header/key column differently
                if key == "Particulars" or key == "Quarter":
                    processed_row[key] = TextProcessor.clean_non_breaking_space(value)
                else:
                    processed_row[key] = TextProcessor.process_table_cell(value)
            processed_data.append(processed_row)
        return processed_data

    @staticmethod
    def clean_currency_values(data: Union[str, int, float, Dict, List]) -> Union[str, int, float, Dict, List]:
        """Clean currency values in nested structures"""
        if isinstance(data, (int, float)):
            return data
        elif isinstance(data, str):
            return data.replace('₹', 'Rs.').replace('\u20b9', 'Rs.')
        elif isinstance(data, list):
            return [TextProcessor.clean_currency_values(item) for item in data]
        elif isinstance(data, dict):
            return {k: TextProcessor.clean_currency_values(v) for k, v in data.items()}
        return data

    @staticmethod
    def extract_json_from_text(text: str, is_financial: bool = False, logger: logging.Logger = None) -> Dict[str, Any]:
        """Extract and parse JSON from response text"""
        try:
            # Clean the text before parsing
            if '```json' in text:
                json_str = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                json_str = text.split('```')[1].split('```')[0].strip()
            else:
                json_str = text.strip()

            # Remove any escaped quotes and extra formatting
            json_str = json_str.replace('\\"', '"').replace('\\n', '')
            if json_str.startswith('"') and json_str.endswith('"'):
                json_str = json_str[1:-1]

            # Parse JSON
            data = json.loads(json_str)
            
            # Clean any nested JSON strings that might be in the data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        try:
                            data[key] = json.loads(value)
                        except:
                            pass

            return TextProcessor.clean_currency_values(data)

        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"JSON parsing error: {e}")
            # Return empty structure based on type
            return TextProcessor._get_empty_structure(is_financial)
        except Exception as e:
            if logger:
                logger.error(f"Text processing error: {e}")
            raise

    @staticmethod
    def _get_empty_structure(is_financial: bool) -> Dict[str, Any]:
        """Get empty data structure based on type"""
        if is_financial:
            return {
                "Business Model & Strategy": [],
                "Financial Performance & Projections": [],
                "Operational Capabilities & Risks": [],
                "Sectorial Analysis": []
            }
        return {"shortDescription": []}
