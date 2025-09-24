
from zenml import pipeline
from steps.rag.retrieve import retrieve_step
from steps.rag.rerank import rerank_step
from steps.rag.answer import answer_step
from contextual_rag.application.rag.rag_model import RagAnswer

@pipeline
def query_pipeline(question: str) -> RagAnswer:
    candidates = retrieve_step(question)
    ranked_context = rerank_step(question, candidates)
    answer = answer_step(ranked_context, question)
    return answer
