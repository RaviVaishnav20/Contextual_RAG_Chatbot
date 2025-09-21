from typing import List, Tuple

from contextual_rag.application.rag.retriever import retrieve_with_llama_index
from contextual_rag.application.rag.reranking import rerank
from contextual_rag.application.rag.generate_answer import generate_answer


def retrieve_step(query: str) -> List[Tuple[str, float, str]]:
    return retrieve_with_llama_index(query)

def rerank_step(query:str, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float]]:
    return rerank(query, candidates)

def answer(ranked_context: List[Tuple[str, float, str]], question: str):
    if not ranked_context:
        return "No relevant context found."
    answer = generate_answer(ranked_context, question)
    return answer


def get_rag_answer(query: str) -> str:
    
    candidates = retrieve_with_llama_index(query)
    ranked_context = rerank_step(query, candidates)
    return answer(ranked_context, query)
