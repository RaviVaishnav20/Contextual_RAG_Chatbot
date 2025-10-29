from typing import List, Tuple
from contextual_rag.application.rag.retriever import retrieve_with_llama_index
from contextual_rag.application.rag.reranking import rerank
from contextual_rag.application.rag.generate_answer import generate_answer
from contextual_rag.utils.save_data import save_rag_response
from contextual_rag.application.rag.rag_model import RetrieverOutput, RerankedOutput, RagAnswer

async def retrieve_step(query: str) -> List[RetrieverOutput]:
    return retrieve_with_llama_index(query)
 
async def rerank_step(query:str, candidates: List[RetrieverOutput]) -> List[RerankedOutput]:
    return rerank(query, candidates)

async def answer(ranked_context: List[RerankedOutput], question: str) -> RagAnswer:
    if not ranked_context:
        return "No relevant context found."
    results = generate_answer(ranked_context, question)
    return results


async def get_rag_answer(query: str) -> RagAnswer:
    
    candidates = await retrieve_step(query)
    ranked_context = await rerank_step(query, candidates)
    results = await answer(ranked_context, query)
    # answer_text = results.answer
    # context = results.retrieved_contexts
    # sources = results.sources
    # # Save to CSV
    # save_rag_response(query, answer_text, "\n\n".join(context), "\n\n".join(sources))
    return results

async def get_relevant_chunks_with_reranking(query: str) -> List[RerankedOutput]:
    
    candidates = await retrieve_with_llama_index(query)
    ranked_context = await rerank_step(query, candidates)
    
    return ranked_context
    