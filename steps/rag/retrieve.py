from typing import List, Tuple
from zenml import step
from contextual_rag.application.rag.retriever import retrieve_with_llama_index
from contextual_rag.application.rag.rag_model import RetrieverOutput
@step(enable_cache=False)
def retrieve_step(query: str) -> List[RetrieverOutput]:
    return retrieve_with_llama_index(query)
   
 