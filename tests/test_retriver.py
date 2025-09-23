from typing import List, Tuple
from contextual_rag.application.rag.retriever import retrieve_with_llama_index

def retrieve_step(query: str) -> List[Tuple[str, float, str]]:
    return retrieve_with_llama_index(query)
   
if __name__ == "__main__":
    query = "Definitions, scope of application, and delegation of powers"
    retrieved_chunks = retrieve_step(query)
    print(retrieved_chunks)