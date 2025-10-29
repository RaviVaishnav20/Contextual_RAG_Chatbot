
import os
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import sqlite3
from datetime import datetime

from contextual_rag.infrastructure.config_manager import ConfigManager

from crewai.memory.storage.interface import Storage



class LocalFAISSStorage(Storage):
    """Custom FAISS-based local storage for short-term memory"""
    
    def __init__(self, storage_path: str = "./crew_memory", dimension: int = 384):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = dimension
        
        # FAISS index files
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.json"
        self.sqlite_db = self.storage_path / "memory_store.db"
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        self.metadata = []
        self.memory_counter = 0
        
        # Initialize SQLite for metadata storage
        self._init_sqlite()
        
        # Load existing data
        self._load_index()
    
    def _init_sqlite(self):
        """Initialize SQLite database for metadata storage"""
        with sqlite3.connect(self.sqlite_db) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    agent_role TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    embedding_hash TEXT
                )
            ''')
            conn.commit()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.memory_counter = len(self.metadata)
                print(f"✅ Loaded {len(self.metadata)} metadata entries")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing index: {e}")
            # Reset if corrupted
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    def save(self, value: str, metadata: Dict[str, Any] = None, agent: str = None):
        """Save content to memory with FAISS indexing"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([value])
            embedding = embedding.astype('float32')
            embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Prepare metadata
            entry_metadata = {
                'id': self.memory_counter,
                'content': value,
                'metadata': metadata or {},
                'agent_role': agent,
                'timestamp': datetime.now().isoformat(),
                'embedding_hash': hash(embedding.tobytes())
            }
            
            self.metadata.append(entry_metadata)
            self.memory_counter += 1
            
            # Save to SQLite
            with sqlite3.connect(self.sqlite_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO memory_entries (content, metadata, agent_role, embedding_hash)
                    VALUES (?, ?, ?, ?)
                ''', (
                    value,
                    json.dumps(metadata) if metadata else None,
                    agent,
                    str(entry_metadata['embedding_hash'])
                ))
                conn.commit()
            
            # Save to disk periodically (every 10 entries)
            if self.memory_counter % 10 == 0:
                self._save_index()
            
            print(f"💭 Saved memory entry {self.memory_counter-1}: {value[:50]}...")
            
        except Exception as e:
            print(f"❌ Error saving to memory: {e}")
    
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search memory using FAISS similarity search"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(limit, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score >= score_threshold:
                    entry = self.metadata[idx].copy()
                    entry['similarity_score'] = float(score)
                    entry['rank'] = i + 1
                    results.append(entry)
            
            print(f"🔍 Found {len(results)} relevant memories for: {query[:30]}...")
            return results
            
        except Exception as e:
            print(f"❌ Error searching memory: {e}")
            return []
    
    def reset(self):
        """Clear all memory data"""
        try:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.memory_counter = 0
            
            # Clear SQLite
            with sqlite3.connect(self.sqlite_db) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM memory_entries')
                conn.commit()
            
            # Remove files
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                
            print("🗑️ Memory reset successfully")
            
        except Exception as e:
            print(f"❌ Error resetting memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics"""
        return {
            'total_entries': self.index.ntotal,
            'metadata_entries': len(self.metadata),
            'storage_path': str(self.storage_path),
            'index_file_size': self.index_file.stat().st_size if self.index_file.exists() else 0,
            'last_save_time': datetime.now().isoformat()
        }
    
    def export_memories(self, filepath: str):
        """Export memories to JSON for backup"""
        export_data = {
            'metadata': self.metadata,
            'stats': self.get_stats(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Exported {len(self.metadata)} memories to {filepath}")


class PickleMemoryStorage(Storage):
    """Simple pickle-based storage for lightweight use cases"""
    
    def __init__(self, storage_path: str = "./crew_memory_pickle"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_path / "memories.pkl"
        self.memories = self._load_memories()
    
    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load memories from pickle file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'rb') as f:
                    memories = pickle.load(f)
                print(f"✅ Loaded {len(memories)} memories from pickle")
                return memories
        except Exception as e:
            print(f"⚠️ Could not load pickle memories: {e}")
        return []
    
    def _save_memories(self):
        """Save memories to pickle file"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memories, f)
            print(f"💾 Saved {len(self.memories)} memories to pickle")
        except Exception as e:
            print(f"❌ Error saving pickle memories: {e}")
    
    def save(self, value: str, metadata: Dict[str, Any] = None, agent: str = None):
        """Save memory entry"""
        entry = {
            'id': len(self.memories),
            'content': value,
            'metadata': metadata or {},
            'agent_role': agent,
            'timestamp': datetime.now().isoformat()
        }
        self.memories.append(entry)
        self._save_memories()
    
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Simple text-based search"""
        query_lower = query.lower()
        results = []
        
        for memory in self.memories:
            content_lower = memory['content'].lower()
            # Simple scoring based on keyword matches
            score = sum(1 for word in query_lower.split() if word in content_lower) / len(query_lower.split())
            
            if score >= score_threshold:
                memory_copy = memory.copy()
                memory_copy['similarity_score'] = score
                results.append(memory_copy)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    def reset(self):
        """Clear all memories"""
        self.memories = []
        if self.memory_file.exists():
            self.memory_file.unlink()
        print("🗑️ Pickle memories reset")