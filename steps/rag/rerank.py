
from typing import List, Tuple
from typing_extensions import Annotated
from zenml import step

from contextual_rag.application.rag.reranking import rerank


@step(enable_cache=False)
def rerank_step(query:str, candidates: List[Tuple[str, float, str, str]]) ->  List[Tuple[Tuple[str, float, str, str], float]]:
    return rerank(query, candidates)
