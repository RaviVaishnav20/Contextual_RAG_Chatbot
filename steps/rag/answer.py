from typing import List, Tuple
from typing_extensions import Annotated
from zenml import step
from contextual_rag.application.rag.rag_model import RerankedOutput, RagAnswer
from contextual_rag.application.rag.generate_answer import generate_answer





@step(enable_cache=False)
def answer_step(ranked_context: List[RerankedOutput], question: str) -> RagAnswer:
    if not ranked_context:
        return RagAnswer(
            answer="No relevant context found.",
            retrieved_contexts=[],
            sources=[]
        )
    response = generate_answer(ranked_context, question)
    
    return response