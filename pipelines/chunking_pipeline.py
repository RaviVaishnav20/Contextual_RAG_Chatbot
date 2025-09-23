
from typing import List
from zenml import pipeline
from steps.preprocess.load_markdowns import load_markdowns
from steps.preprocess.chunk_with_llm import chunk_with_llm_step
from steps.preprocess.chunk_with_context import chunk_with_context_step
from contextual_rag.application.preprocessing.chunking_data_handlers import Chunk
from contextual_rag.settings import settings
import os
import json

@pipeline
def chunking_pipeline():
    files = load_markdowns()
    

    metadata_sementic_chunk_file = settings.metadata_sementic_chunk_file

    # Load existing metadata (best-effort). If invalid JSON, start fresh.
    existing_chunks_json: List[dict] = []
    if os.path.exists(metadata_sementic_chunk_file):
        try:
            with open(metadata_sementic_chunk_file, 'r') as f:
                existing_chunks_json = json.load(f) or []
            print(f"📚 Loaded {len(existing_chunks_json)} existing chunks")
        except Exception:
            print("⚠️ Existing metadata file is invalid JSON. It will be overwritten.")
            existing_chunks_json = []

    # sementic_chunks = [Chunk(c) for c in existing_chunks_json]
    sementic_chunks = [Chunk(
                    document_name=c["document_name"],
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    metadata=c["metadata"],
                ) for c in existing_chunks_json]  
    # sementic_chunks = chunk_with_llm_step(files)   
    contextual_chunks = chunk_with_context_step(files, sementic_chunks)
    return contextual_chunks

