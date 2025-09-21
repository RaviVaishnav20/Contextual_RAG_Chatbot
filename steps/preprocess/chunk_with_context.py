from typing import List
from zenml import step
import json
from dataclasses import asdict
from contextual_rag.application.preprocessing.chunking_data_handlers import build_chunks_context, Chunk
from contextual_rag.infrastructure.materializers import ChunkListMaterializer
from contextual_rag.settings import settings
import os
from pathlib import Path

@step(enable_cache=False, output_materializers=ChunkListMaterializer)
def chunk_with_context_step(files: List[str], sementic_chunks: List[Chunk]) -> List[Chunk]:
    metadata_context_chunk_file = settings.metadata_context_chunk_file

    # Load existing metadata (best-effort). If invalid JSON, start fresh.
    existing_chunks_json: List[dict] = []
    if os.path.exists(metadata_context_chunk_file):
        try:
            with open(metadata_context_chunk_file, 'r') as f:
                existing_chunks_json = json.load(f) or []
            print(f"📚 Loaded {len(existing_chunks_json)} existing chunks")
        except Exception:
            print("⚠️ Existing metadata file is invalid JSON. It will be overwritten.")
            existing_chunks_json = []

    # Build chunks for each file
    all_chunks: List[Chunk] = []
    if files:
        for f in files:
            p = Path(f)
            sementic_chunk = filter_chunks_by_document(sementic_chunks,p.name)
            chunks_with_context= build_chunks_context(p, sementic_chunk)
            # Flatten
            all_chunks.extend(chunks_with_context)

    # Persist JSON-serializable representation
    # Convert dataclasses to dicts and append to any existing JSON entries
    all_chunks_json: List[dict] = existing_chunks_json + [asdict(c) for c in all_chunks]

    # Ensure parent directory exists
    Path(metadata_context_chunk_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_context_chunk_file, 'w') as f:
        json.dump(all_chunks_json, f, indent=4)
    print(f"\n💾 Saved {len(all_chunks_json)} total chunks")

    # Return the structured dataclass list for downstream steps
    return all_chunks

def filter_chunks_by_document(chunks: List[Chunk], doc_name: str) -> List[Chunk]:
    """Return all chunks belonging to a specific document."""
    return [c for c in chunks if c.document_name == doc_name]
