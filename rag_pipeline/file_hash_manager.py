import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os
import json
import hashlib
from typing import Dict, Set
from config.config_manager import ConfigManager

class FileHashManager:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.hash_cache_file = config.get('directories.file_hash_cache')
        self.hash_cache = self._load_hash_cache()
    
    def _load_hash_cache(self) -> Dict[str, str]:
        """Load existing hash cache from file"""
        if os.path.exists(self.hash_cache_file):
            with open(self.hash_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_hash_cache(self):
        """Save hash cache to file"""
        with open(self.hash_cache_file, 'w') as f:
            json.dump(self.hash_cache, f, indent=4)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_changed_or_new_files(self, resources_folder: str) -> Set[str]:
        """Get list of files that are new or have been modified"""
        changed_files = set()
        
        for filename in os.listdir(resources_folder):
            file_path = os.path.join(resources_folder, filename)
            if os.path.isfile(file_path):
                current_hash = self._calculate_file_hash(file_path)
                cached_hash = self.hash_cache.get(filename)
                
                if cached_hash != current_hash:
                    changed_files.add(filename)
                    self.hash_cache[filename] = current_hash
        
        self._save_hash_cache()
        return changed_files
    
    def update_file_hash(self, filename: str, file_path: str):
        """Update hash for a specific file"""
        new_hash = self._calculate_file_hash(file_path)
        self.hash_cache[filename] = new_hash
        self._save_hash_cache()