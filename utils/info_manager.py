from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, List, Optional
import logging

class InfoManager:
    def __init__(self, stock_folder: Path, logger: logging.Logger):
        self.stock_folder = stock_folder
        self.logger = logger
        self.latest_info_dir = stock_folder / "Latest_Info"
        self.latest_info_dir.mkdir(parents=True, exist_ok=True)

    def save_latest_document(self, content_path: Path, doc_type: str, date_str: str) -> Optional[Path]:
        """Save document to Latest_Info directory with date information"""
        try:
            if not content_path or not content_path.exists():
                return None

            # Create target path in Latest_Info with date
            target_path = self.latest_info_dir / f"latest_{doc_type}_{date_str}{content_path.suffix}"
            
            # Copy file
            shutil.copy2(content_path, target_path)
            self.logger.info(f"Saved latest {doc_type} ({date_str}) to {target_path}")
            
            return target_path
            
        except Exception as e:
            self.logger.error(f"Failed to save {doc_type}: {e}")
            return None
