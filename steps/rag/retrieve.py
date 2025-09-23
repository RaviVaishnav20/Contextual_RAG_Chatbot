from typing import List, Tuple
from zenml import step
from contextual_rag.application.rag.retriever import retrieve_with_llama_index

@step(enable_cache=False)
def retrieve_step(query: str) -> List[Tuple[str, float, str]]:
    return retrieve_with_llama_index(query)
   
 