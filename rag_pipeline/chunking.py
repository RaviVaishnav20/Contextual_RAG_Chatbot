import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os
import numpy as np
import requests
import logging
from dotenv import load_dotenv

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm.llm import generate_content
from config.config_manager import ConfigManager

load_dotenv()
logger = logging.getLogger(__name__)

def chunk_text(text: str, config: ConfigManager) -> list[str]:
    """Simple chunking by splitting text into overlapping segments of words."""
    chunking_config = config.get_chunking_config()
    size = chunking_config.get('parameters', {}).get('chunk_size', 256)
    overlap = chunking_config.get('parameters', {}).get('chunk_overlap', 50)
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

def semantic_merge(text: str, config: ConfigManager) -> list[str]:
    """Splits text semantically using LLM: detects second topic and reuses leftover intelligently."""
    chunking_config = config.get_chunking_config()
    WORD_LIMIT = chunking_config.get('parameters', {}).get('word_limit', 512)
    
    primary_provider = chunking_config.get('model', {}).get('provider', 'ollama')
    primary_model = chunking_config.get('model', {}).get('model_name', 'llama3:8b')
    fallback_provider = chunking_config.get('model', {}).get('fallback_provider', 'gemini')
    fallback_model = chunking_config.get('model', {}).get('fallback_model_name', 'gemini-2.5-flash')
    
    words = text.split()
    i = 0
    final_chunks = []

    while i < len(words):
        chunk_words = words[i:i + WORD_LIMIT]
        chunk_text = " ".join(chunk_words).strip()

        prompt = f"""
You are a markdown document segmenter.

Here is a portion of a markdown document:

---
{chunk_text}
---

If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

If it's only one topic, reply with NOTHING.

Keep markdown formatting intact.
"""
        reply = ""
        try:
            reply = generate_content(
                provider=primary_provider,
                model_name=primary_model,
                prompt=prompt
            ).strip()

            if reply:
                split_point = chunk_text.find(reply)

                if split_point != -1:
                    first_part = chunk_text[:split_point].strip()
                    second_part = reply.strip()

                    final_chunks.append(first_part)

                    leftover_words = second_part.split()
                    words = leftover_words + words[i + WORD_LIMIT:]
                    
                    i = 0
                    continue
                else:
                    final_chunks.append(chunk_text)
                    print(f"Chunk: {len(final_chunks)}")
            else:
                final_chunks.append(chunk_text)   
                print(f"Chunk: {len(final_chunks)}") 
        except Exception as e:
            try:
                reply = generate_content(
                    provider=fallback_provider,
                    model_name=fallback_model,
                    prompt=prompt
                ).strip()
                
                if reply:
                    split_point = chunk_text.find(reply)
                    if split_point != -1:
                        first_part = chunk_text[:split_point].strip()
                        second_part = reply.strip()
                        final_chunks.append(first_part)
                        leftover_words = second_part.split()
                        words = leftover_words + words[i + WORD_LIMIT:]
                        i = 0
                        continue
                    else:
                        final_chunks.append(chunk_text)
                        print(f"Chunk: {len(final_chunks)}")
                else:
                    final_chunks.append(chunk_text)
                    print(f"Chunk: {len(final_chunks)}")
            except Exception as fallback_e:
                final_chunks.append(chunk_text)
                print(f"Chunk: {len(final_chunks)}")

        i += WORD_LIMIT

    return final_chunks


def chunk_documents(full_markdown_content: str, filename: str) -> list[str]:
    """Chunk the markdown content using semantic merge"""
    config = ConfigManager()
    chunking_config = config.get_chunking_config()
    
    # Use semantic_merge for chunking
    semantically_merged_chunks = semantic_merge(full_markdown_content, config)
    
    return semantically_merged_chunks

if __name__ == "__main__":
    config = ConfigManager()
    FILE_PATH = config.get('directories.resources') + "/Abu Dhabi Procurement Standards.PDF"
    MARKDOWN_ARTIFACTS_FOLDER = config.get('directories.markdown_artifacts')
    
    os.makedirs(MARKDOWN_ARTIFACTS_FOLDER, exist_ok=True)
    
    # Test markdown conversion
    full_markdown_doc = MARKDOWN_ARTIFACTS_FOLDER+'/Document Q&A Examplary Questions.md'
    
    # Test chunking
    chunks = chunk_documents(full_markdown_doc, "Abu Dhabi Procurement Standards.PDF")