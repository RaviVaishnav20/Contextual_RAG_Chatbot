from dataclasses import dataclass
from typing import Iterable, ClassVar
import os

from loguru import logger
from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from typing import List 
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore

from contextual_rag.settings import settings
from contextual_rag.application.preprocessing.chunking_data_handlers import Chunk
from contextual_rag.infrastructure.llm import get_embedding_batch
from contextual_rag.infrastructure.config_manager import ConfigManager

from llama_index.core.embeddings import BaseEmbedding


class CustomEmbedding(BaseEmbedding):
    cm: ClassVar[ConfigManager] = ConfigManager()
    embed_config: ClassVar[dict] = cm.get_embedding_config() or {}
    provider: ClassVar[str] = embed_config.get('provider', 'ollama')
    model: ClassVar[str] = embed_config.get('model_name', 'nomic-embed-text')
    dimension: ClassVar[int] = embed_config.get('dimension', 768)
    timeout: ClassVar[int] = embed_config.get('timeout', 120)
    
    def _get_text_embedding(self, text: str):
        return get_embedding_batch([text], self.provider, self.model, self.dimension, self.timeout)[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return get_embedding_batch([query], self.provider, self.model, self.dimension, self.timeout)[0]
    
    async def _aget_text_embedding(self, text: str):
        return get_embedding_batch([text], self.provider, self.model, self.dimension, self.timeout)[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return get_embedding_batch([query], self.provider, self.model, self.dimension, self.timeout)[0]


def setup_pgvector_store():
    """Setup PGVector store following official LlamaIndex documentation"""
    cm = ConfigManager()
    db_cfg = cm.get_database_config() or {}
    embed_config = cm.get_embedding_config() or {}
    
    db_host = db_cfg.get('host', 'localhost')
    db_port = db_cfg.get('port', '5432')
    db_name = db_cfg.get('database', 'vector_db')
    db_table_name = db_cfg.get('table_name', 'test_embed')
    db_user = db_cfg.get('user', 'ravi')
    db_password = db_cfg.get('password', 'password')
    db_hnsw_kwargs = db_cfg.get('hnsw_kwargs', {})

    embed_dim = embed_config.get('dimension', 768)

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print("connection_string")
    print(connection_string)
    print(db_table_name)
    # connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    # print(connection_string)
    url = make_url(connection_string)
    
    vector_store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=db_table_name,
        embed_dim=embed_dim,
        hnsw_kwargs=db_hnsw_kwargs,
    )
    
    return vector_store

def create_documents_from_chunks(chunks: List[Chunk]):
    """Convert text chunks to LlamaIndex Documents"""
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc = Document(
            text=chunk.text,
            metadata={"document_name":chunk.document_name, "chunk_id":chunk.chunk_id, "source":chunk.metadata["source"]}
        )
        documents.append(doc)
    
    return documents

def upsert_pgvector(
    chunks: List[Chunk]
):
    """Main function to create embeddings and store in PGVector using official pattern"""

    # embed_model = setup_ollama_embeddings(config)
    
    vector_store = setup_pgvector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = create_documents_from_chunks(chunks)
    
    custom_embed = CustomEmbedding()
    
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        show_progress=True,
        embed_model=custom_embed
    )
    
    return len(chunks)


