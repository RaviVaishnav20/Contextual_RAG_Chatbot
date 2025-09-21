
from zenml import pipeline
from steps.rag.retrieve import retrieve_step
from steps.rag.rerank import rerank_step
from steps.rag.answer import answer_step


@pipeline
def query_pipeline(question: str) -> str:
    candidates = retrieve_step(question)
    ranked_context = rerank_step(question, candidates)
    answer = answer_step(ranked_context, question)
    return answer
