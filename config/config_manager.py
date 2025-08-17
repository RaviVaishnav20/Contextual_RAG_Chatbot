import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = ROOT/"config"/"config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'database.host')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.get('database', {})
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return self.config.get('chunking', {})
    
    def get_contextual_config(self) -> Dict[str, Any]:
        """Get contextual retrieval configuration"""
        return self.config.get('contextual_retrieval', {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.config.get('embedding', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration"""
        return self.config.get('rag', {})
    
    def get_directories(self) -> Dict[str, str]:
        """Get directory paths"""
        return self.config.get('directories', {})