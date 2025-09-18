import sys
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Added {ROOT} to sys.path")

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from config.config_manager import ConfigManager

load_dotenv()

# ----------------------
# Setup Ollama embeddings & LLM
# ----------------------
def setup_ollama(config: ConfigManager):
    embedding_config = config.get_embedding_config()
    embed_model = OllamaEmbedding(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=config.get_ollama_host(),
        ollama_additional_kwargs=embedding_config.get("additional_kwargs", {"mirostat": 0}),
        timeout=120,
    )
    Settings.embed_model = embed_model

    rag_config = config.get_rag_config()
    llm = Ollama(
        model=rag_config.get("model", {}).get("model_name", "gemma3:latest"),
        base_url=config.get_ollama_host(),
        request_timeout=120,
    )
    Settings.llm = llm
    return embed_model, llm

# ----------------------
# Convert chunks to documents
# ----------------------
def create_documents_from_chunks(chunks: List[str], metadata_list: List[dict] = None):
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {"chunk_id": i}
        documents.append(Document(text=chunk, metadata=metadata))
    return documents

# ----------------------
# Simple test script
# ----------------------
if __name__ == "__main__":
    config = ConfigManager()
    embed_model, llm = setup_ollama(config)

    # Test embedding
    test_text = "Hello from LlamaIndex test script!"
    embedding_vector = embed_model.get_text_embedding(test_text)
    print("Embedding vector length:", len(embedding_vector))
    print("First 10 values:", embedding_vector[:10])

    # Test LLM
 # Test LLM
    from llama_index.core.message_schema import ChatMessage, RoleType

    prompt = "Write a short greeting for a user testing LlamaIndex and Ollama integration."

    # Wrap prompt in ChatMessage
    messages = [ChatMessage(role=RoleType.USER, content=prompt)]

    # Pass to llm.chat()
    response = llm.chat(messages)

    print("\nLLM response:\n", response)


    # Optional: create documents and print
    chunks = ["This is chunk 1.", "This is chunk 2."]
    docs = create_documents_from_chunks(chunks)
    print("\nDocuments created:")
    for doc in docs:
        print(doc)
