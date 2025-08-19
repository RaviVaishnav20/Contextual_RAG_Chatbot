import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import asyncio
from rag_pipeline.embedding_storage import load_existing_index, query_embeddings, setup_ollama_embeddings
from llm.llm import generate_content
from config.config_manager import ConfigManager
from agentic_rag.ollama_reranker import OllamaReRanker

async def get_relevant_chunks_with_reranking(user_query: str, initial_k: int = 15, final_k: int = 5):
    """Get chunks with Ollama re-ranking"""
    config = ConfigManager()
    
    setup_ollama_embeddings(config)
    index = load_existing_index(config)
    
    # Step 1: Get more chunks initially
    rag_config = config.get_rag_config()
    original_top_k = rag_config.get('parameters', {}).get('similarity_top_k', 5)
    rag_config['parameters']['similarity_top_k'] = initial_k
    
    response = query_embeddings(index, user_query, config)
    
    # Restore original setting
    rag_config['parameters']['similarity_top_k'] = original_top_k
    
    if not response or not response.source_nodes:
        return {"retrieved_text": ""}
    
    # Step 2: Extract documents
    documents = []
    metadata_list = []
    
    for node in response.source_nodes:
        documents.append(node.text)
        metadata_list.append({
            'source': node.metadata.get('source', 'Unknown'),
            'chunk_id': node.metadata.get('chunk_id', 'N/A')
        })
    
    # Step 3: Re-rank with Ollama
    reranker = OllamaReRanker()
    reranked_results = reranker.rerank(user_query, documents, top_k=final_k)
    
    # Step 4: Format output
    retrieved_text = ""
    for i, (doc, score) in enumerate(reranked_results):
        # Find original metadata
        original_idx = documents.index(doc)
        metadata = metadata_list[original_idx]
        
        retrieved_text += f"--- Chunk {i+1} (Score: {score:.3f}) ---\n"
        retrieved_text += f"Source: {metadata['source']} (Chunk {metadata['chunk_id']})\n"
        retrieved_text += doc + "\n\n"
    
    return {"retrieved_text": retrieved_text}

async def get_rag_response(user_query: str, initial_k: int = 15, final_k: int = 5):
    """Get RAG response with Ollama re-ranking"""
    retrieved_data = await get_relevant_chunks_with_reranking(user_query, initial_k, final_k)
    retrieved_text = retrieved_data["retrieved_text"]

    if not retrieved_text:
        return {"retrieved_text": "", "llm_response": "No relevant context found."}

    prompt = f"Based on the following context, answer the question:\n\nContext:\n{retrieved_text}\n\nQuestion: {user_query}\n\nAnswer:"
    
    config = ConfigManager()
    rag_config = config.get_rag_config()
    primary_model = rag_config.get('model', {})
    
    try:
        llm_response = generate_content(
            provider=primary_model.get('provider', 'ollama'),
            model_name=primary_model.get('model_name', 'llama3:8b'),
            prompt=prompt
        )
    except Exception as e:
        try:
            llm_response = generate_content(
            provider=primary_model.get('fallback_provider', 'ollama'),
            model_name=primary_model.get('fallback_model_name', 'llama3:8b'),
            prompt=prompt
        )
        except Exception as e:
            llm_response = f"Error: {str(e)}"
    
    return {"retrieved_text": retrieved_text, "llm_response": llm_response}

async def main():
    user_query = input("\nEnter your query: ")
    result = await get_rag_response(user_query)
    print("\nRe-ranked Results:")
    print(result["retrieved_text"])
    print("\nLLM Answer:")
    print(result["llm_response"])

if __name__ == "__main__":
    asyncio.run(main())