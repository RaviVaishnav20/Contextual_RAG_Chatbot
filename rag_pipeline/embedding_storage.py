import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Added {ROOT} to sys.path") # Debug print
    
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, make_url
from llama_index.llms.ollama import Ollama 
from config.config_manager import ConfigManager
import os
from dotenv import load_dotenv
load_dotenv()
def setup_ollama_embeddings(config: ConfigManager):
    """Initialize Ollama embedding model with configuration"""
    embedding_config = config.get_embedding_config()
    
    embed_model = OllamaEmbedding(
        model_name=embedding_config.get('model_name', 'nomic-embed-text'),
        base_url=config.get_ollama_host(),
        ollama_additional_kwargs=embedding_config.get('additional_kwargs', {"mirostat": 0}),
        timeout=120
    )
    
    Settings.embed_model = embed_model

    rag_config = config.get_rag_config()
    llm = Ollama(
        model=rag_config.get('model', {}).get('model_name', 'llama3:8b'), 
        base_url=config.get_ollama_host(),
        request_timeout=120
    )
    Settings.llm = llm
    return embed_model

def setup_pgvector_store(config: ConfigManager):
    """Setup PGVector store following official LlamaIndex documentation"""
    db_config = config.get_database_config()
    embedding_config = config.get_embedding_config()
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name =os.getenv("DATABASE_NAME", "vector_db")
    db_table_name =os.getenv("DATABASE_TABLE_NAME", "contextual_embedding")
    db_user =os.getenv("DATABASE_USER", "ravi")
    db_password = os.getenv("DATABASE_PASSWORD", "password")


    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    # print(connection_string)
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
        embed_dim=embedding_config.get('dimension', 768),
        hnsw_kwargs=db_config.get('hnsw_kwargs', {}),
    )
    
    return vector_store


    

def create_documents_from_chunks(chunks: List[str], metadata_list: List[dict] = None):
    """Convert text chunks to LlamaIndex Documents"""
    documents = []
    
    for i, chunk in enumerate(chunks):
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {"chunk_id": i}
        
        doc = Document(
            text=chunk,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

async def process_and_store_embeddings(
    chunks: List[str],
    metadata_list: List[dict] = None,
    db_config: dict = None,
    config: ConfigManager = None
):
    """Main function to create embeddings and store in PGVector using official pattern"""
    
    if config is None:
        config = ConfigManager()
    
    embed_model = setup_ollama_embeddings(config)
    
    vector_store = setup_pgvector_store(config)
    
    documents = create_documents_from_chunks(chunks, metadata_list)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        show_progress=True
    )
    
    return index, vector_store

def query_embeddings(index: VectorStoreIndex, query: str, config: ConfigManager = None):
    """Query the vector store using official LlamaIndex pattern"""
    if config is None:
        config = ConfigManager()
    
    rag_config = config.get_rag_config()
    top_k = rag_config.get('parameters', {}).get('similarity_top_k', 5)
    
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return response

def load_existing_index(config: ConfigManager = None):
    """Load an existing index from PGVector storage"""
    if config is None:
        config = ConfigManager()
    
    setup_ollama_embeddings(config)
    vector_store = setup_pgvector_store(config)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex([], storage_context=storage_context)
    return index

if __name__=="__main__":
    user_query = "Benifit tracking"
    config = ConfigManager()
    
    initial_k = 15
    setup_ollama_embeddings(config)
    index = load_existing_index(config)
    
    # Step 1: Get more chunks initially
    rag_config = config.get_rag_config()
    original_top_k = rag_config.get('parameters', {}).get('similarity_top_k', 5)
    rag_config['parameters']['similarity_top_k'] = initial_k
    
    response = query_embeddings(index, user_query, config)
    print(response)
    
    