
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
    artifacts_dir: Path = data_dir / "artifacts"
    metadata_sementic_chunk_file = artifacts_dir / "metadata_sementic_chunk.json"
    metadata_context_chunk_file = artifacts_dir / "metadata_context_chunk.json"
    chunk_metadata_file = metadata_context_chunk_file
    crew_dir = project_root/"contextual_rag"/"application"/"agents"/"crew"

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
