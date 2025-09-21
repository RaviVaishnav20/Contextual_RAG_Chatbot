"""Custom materializers for ZenML artifacts."""

import json
from typing import Any, Type, Union
from pathlib import Path

from zenml.materializers.base_materializer import BaseMaterializer
from contextual_rag.application.preprocessing.chunking_data_handlers import Chunk


class ChunkMaterializer(BaseMaterializer):
    """Custom materializer for Chunk objects using JSON format."""
    
    ASSOCIATED_TYPES = (Chunk,)
    
    def load(self, data_type: Type[Any]) -> Chunk:
        """Load a Chunk object from JSON file."""
        file_path = Path(self.uri) / "data.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return Chunk(
            document_name=data["document_name"],
            chunk_id=data["chunk_id"],
            text=data["text"],
            metadata=data["metadata"]
        )
    
    def save(self, chunk: Chunk) -> None:
        """Save a Chunk object to JSON file."""
        file_path = Path(self.uri) / "data.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "document_name": chunk.document_name,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class ChunkListMaterializer(BaseMaterializer):
    """Custom materializer for List[Chunk] objects using JSON format."""
    
    ASSOCIATED_TYPES = (list,)
    
    def load(self, data_type: Type[Any]) -> list[Chunk]:
        """Load a list of Chunk objects from JSON file."""
        file_path = Path(self.uri) / "data.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [
            Chunk(
                document_name=chunk_data["document_name"],
                chunk_id=chunk_data["chunk_id"],
                text=chunk_data["text"],
                metadata=chunk_data["metadata"]
            )
            for chunk_data in data
        ]
    
    def save(self, chunks: list[Chunk]) -> None:
        """Save a list of Chunk objects to JSON file."""
        file_path = Path(self.uri) / "data.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                "document_name": chunk.document_name,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
