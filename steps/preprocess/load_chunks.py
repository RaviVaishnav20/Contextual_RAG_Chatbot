import os
import json
from pathlib import Path
from typing import List
from typing_extensions import Annotated
from zenml import step
from contextual_rag.application.preprocessing.chunking_data_handlers import Chunk
from contextual_rag.settings import settings
#note: todo file hashing

@step(enable_cache=False)
def load_chunks() -> List[Chunk]:
    chunk_metadata_file = settings.chunk_metadata_file
    # Load existing metadata (best-effort). If invalid JSON, start fresh.
    chunks = []
    # existing_chunks_json: List[dict] = []
    if os.path.exists(chunk_metadata_file):
        try:
            with open(chunk_metadata_file, 'r') as f:
                existing_chunks_json = json.load(f) or []
            print(f"📚 Loaded {len(existing_chunks_json)} existing chunks")
            if len(existing_chunks_json) > 0:
                chunks = [Chunk(
                        document_name=c["document_name"],
                        chunk_id=c["chunk_id"],
                        text=c["text"],
                        metadata=c["metadata"],
                    ) for c in existing_chunks_json]  
            else:
                chunks = []
        except Exception:
            print("⚠️ Existing metadata file is invalid JSON. It will be overwritten.")
            chunks = []
        
    return chunks
