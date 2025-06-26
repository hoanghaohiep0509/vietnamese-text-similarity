import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations for the text similarity project"""
    
    @staticmethod
    def read_csv(file_path: str) -> Optional[pd.DataFrame]:
        """Read CSV file and return DataFrame"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Successfully read CSV file: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_csv(data: List[Dict[str, Any]], file_path: str) -> bool:
        """Write data to CSV file"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Successfully wrote CSV file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing CSV file {file_path}: {e}")
            return False
    
    @staticmethod
    def read_json(file_path: str) -> Optional[Dict[str, Any]]:
        """Read JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully read JSON file: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: str) -> bool:
        """Write data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully wrote JSON file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def append_to_csv(data: Dict[str, Any], file_path: str) -> bool:
        """Append a single row to CSV file"""
        try:
            file_exists = os.path.exists(file_path)
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(data)
            
            logger.info(f"Successfully appended to CSV file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error appending to CSV file {file_path}: {e}")
            return False