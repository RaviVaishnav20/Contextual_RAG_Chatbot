from typing import List
from zenml import step
import json
from dataclasses import asdict
from contextual_rag.application.preprocessing.chunking_data_handlers import build_chunks, Chunk
from contextual_rag.infrastructure.materializers import ChunkListMaterializer
from contextual_rag.settings import settings
import os
from pathlib import Path

@step(output_materializers=ChunkListMaterializer)
def chunk_with_llm_step(files: List[str]) -> List[Chunk]:
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

    # Build chunks for each file
    all_chunks: List[Chunk] = []
    if files:
        for f in files:
            p = Path(f)
            chunks_for_file = build_chunks(p)
            # Flatten
            all_chunks.extend(chunks_for_file)

    # Persist JSON-serializable representation
    # Convert dataclasses to dicts and append to any existing JSON entries
    all_chunks_json: List[dict] = existing_chunks_json + [asdict(c) for c in all_chunks]

    # Ensure parent directory exists
    Path(metadata_sementic_chunk_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_sementic_chunk_file, 'w') as f:
        json.dump(all_chunks_json, f, indent=4)
    print(f"\n💾 Saved {len(all_chunks_json)} total chunks")

    # Return the structured dataclass list for downstream steps
    return all_chunks
