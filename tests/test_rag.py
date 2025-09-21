from typing import List, Tuple

from contextual_rag.application.rag.retriever import retrieve_with_llama_index
from contextual_rag.application.rag.reranking import rerank
from contextual_rag.application.rag.generate_answer import generate_answer


def retrieve_step(query: str) -> List[Tuple[str, float, str]]:
    return retrieve_with_llama_index(query)

def rerank_step(query:str, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float]]:
    return rerank(query, candidates)

def test_answer(ranked_context: List[Tuple[str, float, str]], question: str):
    if not ranked_context:
        return "No relevant context found."
    answer = generate_answer(ranked_context, question)
    
    return answer

if __name__=="__main__":
    query = "Definitions, scope of application, and delegation of powers"
    candidates = retrieve_with_llama_index(query)
    ranked_context = rerank_step(query, candidates)

    question = "Definitions, scope of application, and delegation of powers"
    test_answer(ranked_context, question)