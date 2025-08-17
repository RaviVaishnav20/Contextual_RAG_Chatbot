import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Added {ROOT} to sys.path") # Debug print
import asyncio
from rag_pipeline.embedding_storage import load_existing_index, query_embeddings, setup_ollama_embeddings
from llm.llm import generate_content
from config.config_manager import ConfigManager

async def get_relevant_chunks(user_query: str):
    config = ConfigManager()
    
    setup_ollama_embeddings(config)
    
    index = load_existing_index(config)
    
    response = query_embeddings(index, user_query, config)
    
    retrieved_text = ""
    if response and response.source_nodes:
        for i, node in enumerate(response.source_nodes):
            retrieved_text += node.text + "\n"
    return {"retrieved_text": retrieved_text}


async def get_rag_response(user_query: str):
    retrieved_data = await get_relevant_chunks(user_query)
    retrieved_text = retrieved_data["retrieved_text"]

    if not retrieved_text:
        return {"retrieved_text": "", "llm_response": "No relevant context found."}

    llm_response = "No answer generated."
    prompt = f"Based on the following context, answer the question:\n\nContext:\n{retrieved_text}\n\nQuestion: {user_query}\n\nAnswer:"
    
    config = ConfigManager()
    rag_config = config.get_rag_config()
    primary_model = rag_config.get('model', {})
    fallback_model = rag_config.get('model', {})
    
    try:
        llm_response = generate_content(
            provider=primary_model.get('provider', 'ollama'),
            model_name=primary_model.get('model_name', 'llama3.1:8b'),
            prompt=prompt
        )
    except Exception as e:
        llm_response = generate_content(
            provider=fallback_model.get('fallback_provider', 'gemini'),
            model_name=fallback_model.get('fallback_model_name', 'gemini-2.5-flash'),
            prompt=prompt
        )
    return {"retrieved_text": retrieved_text, "llm_response": llm_response}


async def main():
    user_query = input("\nEnter your query: ")
    result = await get_rag_response(user_query)
    print("\nRetrieved Text:")
    print(result["retrieved_text"])
    print("\nLLM Answer:")
    print(result["llm_response"])

if __name__ == "__main__":
    asyncio.run(main())