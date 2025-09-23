from zenml import pipeline
from steps.preprocess.load_chunks import load_chunks
from steps.preprocess.embed_chunks import embed_chunks_step


@pipeline
def build_embedding():
    chunks = load_chunks()
    vector_stats = embed_chunks_step(chunks)
    return vector_stats
