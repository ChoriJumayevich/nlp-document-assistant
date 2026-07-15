import json
from typing import List, Dict
from loguru import logger

class KBLoader:
    """Utility to load the Uzbek Knowledge Base from JSONL format."""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Loads a JSONL file and returns a list of dictionaries."""
        data = []
        logger.info(f"Loading knowledge base from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            logger.info(f"Successfully loaded {len(data)} entries.")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading JSONL: {e}")
            
        return data

kb_loader = KBLoader()
