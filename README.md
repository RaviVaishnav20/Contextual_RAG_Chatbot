# Contextual RAG Chatbot

> **Agentic contextual RAG pipeline**: ingest resources → markdown → chunk → embed → pgvector → retrieve → rerank → answer, with ZenML pipelines.

An advanced RAG (Retrieval-Augmented Generation) chatbot that uses contextual chunking, semantic embeddings, and agentic workflows to provide accurate, context-aware responses from your documents.

## 🚀 Features

- **Contextual Chunking**: Intelligent document chunking with context preservation
- **Semantic Embeddings**: Vector-based document retrieval using pgvector
- **Agentic Workflows**: CrewAI-powered intelligent response generation
- **MLOps Pipeline**: ZenML orchestration for reproducible ML workflows
- **Real-time Tracking**: Phoenix observability for monitoring
- **FastAPI Integration**: RESTful API endpoints for easy integration
- **OpenWebUI**: Modern chat interface for user interaction

## 📋 Prerequisites

- Python 3.13+
- PostgreSQL with pgvector extension
- Docker (for observability and UI)
- uv package manager

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/RaviVaishnav20/Contextual_RAG_Chatbot.git
cd Contextual_RAG_Chatbot
```

### 2. Environment Setup

Create your environment file:

```bash
cp env.example .env
```

Add the following to your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
CREWAI_TRACING_ENABLED=true
```

## 🏗️ Installation & Setup

### Phase 1: ZenML Pipeline Setup

> **Note**: There's a dependency conflict between ZenML and CrewAI, so we'll use separate pyproject.toml files for different phases.

#### Step 1: Initialize ZenML

```bash
# Use ZenML pyproject.toml (included in repo)
uv sync

# Initialize ZenML
uv run zenml init
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local
```

**Troubleshooting ZenML Setup:**
If you encounter port issues:

```bash
rm -rf "/Users/ravi/Library/Application Support/zenml/zen_server"
lsof -i :8237
kill -9 <PID>
```

#### Step 2: Install PostgreSQL with pgvector

**Option 1: Install with Homebrew (Recommended)**

```bash
brew install postgresql@15 pgvector
```

**Option 2: Build pgvector from Source**

```bash
brew install make gcc
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

#### Step 3: Configure PostgreSQL

```bash
# Connect to PostgreSQL
psql -U ravi -d vector_db

# Enable vector extension
CREATE EXTENSION vector;
\dx
```

**Troubleshooting PostgreSQL:**

1. Check if psql is installed:
```bash
which psql
```

2. If missing, add Postgres to PATH:
```bash
nano ~/.zshrc
# Add this line:
export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"
source ~/.zshrc
psql --version
```

3. Connect explicitly:
```bash
psql -U ravi -d postgres -h localhost
psql -U ravi -d vector_db -h localhost
```

### Phase 2: Data Processing Pipeline

#### Step 1: Ingest Resources

Convert documents to markdown using Langchain Docling:

```bash
uv run python -m tools.run ingest
```

This processes files from the `resources/` folder and saves markdown to `data/markdown/`.

#### Step 2: Contextual Chunking

Generate semantic chunks with context:

```bash
uv run python -m tools.run chunking
```

This creates:
- `metadata_sementic_chunk.json`: Initial semantic chunks
- `metadata_context_chunk.json`: Contextually enriched chunks

#### Step 3: Build Embeddings

Create and store vector embeddings in pgvector:

```bash
uv run python -m tools.run embedding
```

#### Step 4: Verify Database Setup

```bash
psql -U ravi -d vector_db
\dt
```

#### Step 5: Test Sample Query

```bash
uv run python -m tools.run query "specifically in Article (137)"
```

### Phase 3: Agentic Workflows & API

#### Step 1: Switch to CrewAI Environment

Replace the pyproject.toml with CrewAI version and sync:

```bash
# Replace pyproject.toml with CrewAI version (included in repo)
uv sync
```

#### Step 2: Setup Phoenix Observability

Start Phoenix container for real-time tracking:

```bash
docker run -d \
  -p 9090:9090 \
  -p 6006:6006 \
  -p 4317:4317 \
  --name phoenix \
  --restart always \
  arizephoenix/phoenix:latest
```

Verify Phoenix is running:
```bash
docker ps --filter "name=phoenix"
```

Access Phoenix at: http://localhost:6006

#### Step 3: Start FastAPI Server

Launch the API endpoints:

```bash
uv run -m contextual_rag.model.inference.api.main
```

Access Swagger documentation at: http://localhost:8000

#### Step 4: Setup OpenWebUI

Setup the chat interface:

```bash
# Stop existing container if running
docker stop open-webui && docker rm open-webui

# Start OpenWebUI
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy-key \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Access OpenWebUI at: http://localhost:3000

## 📁 Project Structure

```
Contextual_RAG_Chatbot/
├── configs/                    # Configuration files
├── contextual_rag/
│   ├── application/           # Application layer
│   │   ├── agents/           # CrewAI agents and tools
│   │   ├── extractors/       # Document extraction
│   │   ├── preprocessing/    # Data preprocessing
│   │   └── rag/             # RAG implementation
│   ├── infrastructure/       # Infrastructure components
│   └── model/               # ML models and API
├── data/
│   ├── artifacts/           # Generated chunks
│   └── markdown/           # Processed documents
├── pipelines/              # ZenML pipelines
├── resources/              # Source documents
├── steps/                  # Pipeline steps
└── tools/                  # Utility scripts
```

## 🔧 Configuration

### Database Configuration (configs/config.yaml)
Configure your PostgreSQL connection settings in the config file.

### Agent Configuration (contextual_rag/application/agents/crew/config/)
- `agents.yaml`: Define AI agents and their roles
- `tasks.yaml`: Configure agent tasks and workflows

## 🧪 Testing

Run various tests to verify functionality:

```bash
# Test basic functionality
python tests/simple_run.py

# Test RAG components
python tests/test_rag.py

# Test document conversion
python tests/test_document_conversion.py

# Test evaluation metrics
python tests/test_ragas.py
```

## 📊 Evaluation

The system includes RAGAS evaluation metrics for assessing RAG performance:

```bash
# Run evaluation
python -m contextual_rag.model.evaluation.ragas
```

## 🔍 Monitoring & Observability

- **Phoenix**: Real-time tracing and monitoring at http://localhost:6006
- **ZenML Dashboard**: Pipeline tracking and artifact management
- **FastAPI Metrics**: API performance monitoring

## 🚀 Usage Examples

### API Usage

```python
import requests

# Query the RAG system
response = requests.post(
    "http://localhost:8000/rag/query",
    json={"query": "What are the HR policies regarding leave?"}
)
print(response.json())
```

### Direct Python Usage

```python
from contextual_rag.application.rag.rag import RAG

rag = RAG()
answer = rag.answer_question("What are the company bylaws?")
print(answer)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **ZenML Server Issues**: Clear server data and restart
2. **PostgreSQL Connection**: Verify credentials and pgvector extension
3. **Docker Containers**: Check port availability and container status
4. **Dependency Conflicts**: Use appropriate pyproject.toml for each phase

### Getting Help

- Check the troubleshooting sections above
- Review test files for usage examples
- Open an issue on GitHub for bugs or feature requests

## 🔗 Related Documentation

- [OpenWebUI Integration Guide](OpenWebUI_and_FastAPI_intergration.docx)
- [ZenML Documentation](https://docs.zenml.io)
- [CrewAI Documentation](https://docs.crewai.com)
- [pgvector Documentation](https://github.com/pgvector/pgvector)