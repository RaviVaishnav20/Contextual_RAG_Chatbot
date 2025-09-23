# Contextual RAG Chatbot

Agentic contextual RAG pipeline: ingest resources → markdown → chunk → embed → pgvector → retrieve → rerank → answer, with ZenML pipelines.

## Quickstart

1. Install: `uv sync`
2. Start services: `docker compose up -d`
3. Ingest resources to Markdown: `uv run python -m tools.run ingest`
4. Build index (chunks + embeddings to pgvector): `uv run python -m tools.run index`
5. Query: `uv run python -m tools.run query "What is in the docs?"`

## Project Structure

```
Contextual_RAG_Chatbot_New/
├── configs/                 # YAML configs
├── data/
│   ├── raw/                # Original files
│   ├── markdown/           # Docling outputs
│   └── artifacts/          # JSON, metrics
├── contextual_rag/         # Main package
│   ├── domain/             # Core entities
│   ├── application/        # Feature modules
│   │   ├── extractors/     # Docling extraction
│   │   ├── preprocessing/  # Chunking, embeddings
│   │   ├── rag/           # Retriever, reranker
│   │   ├── agents/        # CrewAI agents
│   │   └── networks/      # Model clients
│   ├── infrastructure/     # DB, observability
│   └── model/             # Inference, evaluation
├── pipelines/             # ZenML pipelines
├── steps/                 # ZenML steps
├── tools/                 # CLI tools
└── docker-compose.yml     # Postgres+pgvector, Phoenix
```

## Development

- Install dev dependencies: `uv sync --extra dev`
- Run linting: `uv run ruff check .`
- Run tests: `uv run pytest`

## Configuration

### Database Configuration

The project supports flexible PostgreSQL configuration through YAML files and environment variables:

1. **Environment-based config**: Set `ENVIRONMENT` env var to use different configs from `configs/database.yaml`
   - `default`: Local development with Docker Compose
   - `development`: Local development database
   - `production`: Production database with env var substitution
   - `docker`: Docker Compose environment

2. **Direct DSN override**: Set `POSTGRES_DSN` env var to override YAML config

3. **Environment variables**: Copy `env.example` to `.env` and configure as needed

### Example Usage

```bash
# Use default config (Docker Compose)
ENVIRONMENT=default

# Use production config with env vars
ENVIRONMENT=production
DATABASE_HOST=your-host
DATABASE_NAME=your-db
DATABASE_USER=your-user
DATABASE_PASSWORD=your-password
DATABASE_TABLE_NAME=your-table

# Direct DSN override
POSTGRES_DSN=postgresql+psycopg://user:pass@host:port/db?sslmode=require
```

## Notes

- Replace embedding and LLM stubs with your providers.
- Phoenix and RAGAS integration are scaffolded for extension.
