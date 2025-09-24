from typing import List, Tuple
from typing_extensions import Annotated
from zenml import step
from contextual_rag.application.rag.generate_answer import generate_answer





@step(enable_cache=False)
def answer_step(ranked_context: List[Tuple[Tuple[str, float, str, str], float]], question: str) -> str:
    if not ranked_context:
        return "No relevant context found."
    response = generate_answer(ranked_context, question)
    
    return response[0]