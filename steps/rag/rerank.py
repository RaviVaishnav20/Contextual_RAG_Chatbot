
from typing import List, Tuple
from typing_extensions import Annotated
from zenml import step
from contextual_rag.application.rag.rag_model import RetrieverOutput,RerankedOutput
from contextual_rag.application.rag.reranking import rerank


@step(enable_cache=False)
def rerank_step(query:str, candidates: List[RetrieverOutput]) ->  List[RerankedOutput]:
    return rerank(query, candidates)
