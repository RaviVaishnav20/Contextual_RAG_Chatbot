from typing import List, Tuple
from contextual_rag.application.rag.retriever import retrieve_with_llama_index
from contextual_rag.application.rag.reranking import rerank
from contextual_rag.application.rag.generate_answer import generate_answer
from contextual_rag.utils.save_data import save_rag_response

async def retrieve_step(query: str) -> List[Tuple[str, float, str]]:
    return retrieve_with_llama_index(query)
 
async def rerank_step(query:str, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float]]:
    return rerank(query, candidates)

async def answer(ranked_context: List[Tuple[str, float, str]], question: str) ->Tuple[str, List[str]]:
    if not ranked_context:
        return "No relevant context found."
    results = generate_answer(ranked_context, question)
    return results


async def get_rag_answer(query: str) -> Tuple[str, List[str]]:
    
    candidates = await retrieve_step(query)
    ranked_context = await rerank_step(query, candidates)
    results = await answer(ranked_context, query)
    answer_text = results[0]
    context = results[1]
    # Save to CSV
    save_rag_response(query, answer_text, "\n\n".join(context))
    return (answer_text, context)

async def get_relevant_chunks_with_reranking(query: str) -> List[Tuple[str, float]]:
    
    candidates = await retrieve_with_llama_index(query)
    ranked_context = await rerank_step(query, candidates)
    
    return ranked_context
    