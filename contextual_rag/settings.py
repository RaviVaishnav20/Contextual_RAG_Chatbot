
import os
from dataclasses import dataclass
from pathlib import Path

from contextual_rag.infrastructure.config import config_loader


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    markdown_dir: Path = data_dir / "markdown"
    evaluation_dir: Path = data_dir / "evaluation"
    artifacts_dir: Path = data_dir / "artifacts"
    crew_memory_dir: Path = data_dir / "crew_memory"
  
    metadata_sementic_chunk_file = artifacts_dir / "metadata_sementic_chunk.json"
    metadata_context_chunk_file = artifacts_dir / "metadata_context_chunk.json"
    chunk_metadata_file = metadata_context_chunk_file
    rag_metadata_file = artifacts_dir / "rag_metadata_file.csv"
    crew_dir = project_root/"contextual_rag" / "application"/"agents"/"crew"
    ragas_evaluation_report = artifacts_dir / "ragas_evaluation_report.xlsx"
    ragas_metadata_file = artifacts_dir / "ragas_metadata_file.csv"
    ragas_gt_dataset = evaluation_dir/"ragas_gt_data.json"
    ragas_gt_dataset_with_response = evaluation_dir/"ragas_gt_dataset_with_response.csv"
    temp_ragas_gt_dataset = evaluation_dir/"temp_ragas_gt_data.json"
    AGENTIC_RAG_TIMEOUT = 480
    RAG_TIMEOUT = 300
    faiss_index= crew_memory_dir / "faiss_index"
    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "nomic-ai/nomic-embed-text-v1.5" #"sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"
    TEXT_GENERATION_MODEL_ID = "Qwen/Qwen3-1.7B"
    # Database configuration - uses new config system with fallback
    postgres_dsn: str = (
        os.getenv("POSTGRES_DSN") or 
        config_loader.get_postgres_dsn(os.getenv("ENVIRONMENT", "default"))
    )
    database_table_name: str = (
        os.getenv("DATABASE_TABLE_NAME") or
        config_loader.load_database_config(os.getenv("ENVIRONMENT", "default")).table_name
    )
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


settings = Settings()
