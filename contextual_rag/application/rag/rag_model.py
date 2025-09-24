from typing import List
from pydantic import BaseModel

class RetrieverOutput(BaseModel):
    """
    Pydantic model for a single retrieval result.
    """
    chunk_id: str
    score: float
    chunk_content: str
    source: str
    # chunk_context: str

class RerankedOutput(BaseModel):
    """
    Pydantic model for a reranked retrieval result.
    """
    retriever_output: RetrieverOutput
    rerank_score: float

class RagAnswer(BaseModel):
    """
    Pydantic model for a single retrieval result.
    """
    answer: str
    retrieved_contexts: List[str]
    sources: List[str]


 