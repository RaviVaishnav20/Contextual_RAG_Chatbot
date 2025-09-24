from typing import List, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage.storage_context import StorageContext
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.application.preprocessing.embedding_data_handlers import CustomEmbedding, setup_pgvector_store   
from contextual_rag.application.rag.rag_model import RetrieverOutput

def retrieve_with_llama_index(query_text: str) -> List[RetrieverOutput]:
    """
    Retrieve the most relevant chunks using LlamaIndex + PGVectorStore.
    Returns list of (chunk_id, score, text)
    """
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    similarity_top_k = rag_cfg.get('retriver', {}).get('similarity_top_k','')
    
    
    # 1️⃣ Setup PGVectorStore
    vector_store = setup_pgvector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2️⃣ Create an index (no need to rebuild)
    custom_embed = CustomEmbedding()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=custom_embed
    )

    # 3️⃣ Create a retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k
    )

    # 4️⃣ Query the retriever
    results = retriever.retrieve(query_text)
    output = []
    for node in results:
        retriever_output = RetrieverOutput(
            chunk_id=node.metadata.get("chunk_id", ""),
            score=float(node.score),
            chunk_content=node.text,
            source=node.metadata.get("document_name", "")
        )
        output.append(retriever_output)

    return output

    ## If chunk context is available
    # output = []
    # for node in results:
    #     retriever_output = RetrieverOutput(
    #         chunk_id=node.metadata.get("chunk_id", ""),
    #         score=float(node.score),
    #         chunk_content=node.text,
    #         source=node.metadata.get("document_name", ""),
    #         chunk_context=node.metadata.get("chunk_context", ""),
    #     )
    #     output.append(retriever_output)

    # return output
    


        
