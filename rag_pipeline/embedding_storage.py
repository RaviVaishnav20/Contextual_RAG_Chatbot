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

def setup_ollama_embeddings(config: ConfigManager):
    """Initialize Ollama embedding model with configuration"""
    embedding_config = config.get_embedding_config()
    
    embed_model = OllamaEmbedding(
        model_name=embedding_config.get('model_name', 'nomic-embed-text'),
        base_url=embedding_config.get('base_url', 'http://localhost:11434'),
        ollama_additional_kwargs=embedding_config.get('additional_kwargs', {"mirostat": 0}),
    )
    
    Settings.embed_model = embed_model

    rag_config = config.get_rag_config()
    llm = Ollama(
        model=rag_config.get('model', {}).get('model_name', 'llama3.1:8b'), 
        base_url=embedding_config.get('base_url', 'http://localhost:11434')
    )
    Settings.llm = llm
    return embed_model

def setup_pgvector_store(config: ConfigManager):
    """Setup PGVector store following official LlamaIndex documentation"""
    db_config = config.get_database_config()
    embedding_config = config.get_embedding_config()
    
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    url = make_url(connection_string)
    
    vector_store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=db_config['table_name'],
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