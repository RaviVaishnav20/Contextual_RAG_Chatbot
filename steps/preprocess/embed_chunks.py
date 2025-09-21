
from typing import List
from typing_extensions import Annotated
from zenml import step

from contextual_rag.application.preprocessing.chunking_data_handlers import Chunk
from contextual_rag.application.preprocessing.embedding_data_handlers import upsert_pgvector


@step
def embed_chunks_step(chunks: List[Chunk]) -> int:
    return upsert_pgvector(chunks)
