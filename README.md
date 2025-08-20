# Contextual RAG Chatbot

## Installation

Follow these steps to set up the project locally.

### 1. Set up Python Environment

First, ensure you have `uv` installed. If not, you can install it via `pip`:
```bash
pip install uv
```

Then synchronize your project dependencies:
```bash
uv sync
```

### 2. Set PYTHONPATH
```bash
pwd
```
Ensure your Python path is correctly set. Replace `/Users/ravi/Contextual_RAG_ChatBot` with your actual project path.
```bash
export PYTHONPATH=$PYTHONPATH:/Users/ravi/Contextual_RAG_ChatBot
```

### 3. Install PostgreSQL and PGVector

You have two options for installing PostgreSQL and PGVector:

#### Option 1: Install with Homebrew (Recommended)

```bash
brew install postgresql@15 pgvector
```

#### Option 2: Build PGVector from Source

If you need to keep your current Postgres untouched:

```bash
brew install make gcc
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```
This will install `vector.control` and `.sql` files into your Postgres extension directory (e.g., `/opt/homebrew/opt/postgresql@15/share/postgresql@15/extension/`).

### 4. After PostgreSQL Installation

Reconnect to PostgreSQL and enable the `vector` extension.

```bash
psql -U ravi -d vector_db
CREATE EXTENSION vector;
\dx
```

### 5. Troubleshooting PostgreSQL Installation

If you encounter issues, follow these steps:

#### 1. Check if `psql` is installed

Run:
```bash
which psql
```
If it prints something like `/opt/homebrew/opt/postgresql@15/bin/psql`, it's installed correctly. If it prints nothing, your `PATH` doesn't include Postgres.

#### 2. If missing, add Postgres to PATH

Since your installation lives in `/opt/homebrew/opt/postgresql@15`, add this to your `~/.zshrc`:

```bash
nano ~/.zshrc
```
Add this line at the bottom:
```bash
export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"
```
Save and exit, then reload your shell:
```bash
source ~/.zshrc
```
Verify:
```bash
psql --version
```
You should now see something like: `psql (PostgreSQL) 15.x`.

#### 3. Connect Explicitly

Instead of relying on defaults, specify everything:
```bash
psql -U ravi -d postgres -h localhost
```

#### 4. Connect to `vector_db`

```bash
psql -U ravi -d vector_db -h localhost
```

### 6. Create Embeddings

Assuming `metadata_context_chunk.json` is already generated, you can directly create embeddings:
```bash
uv run rag_pipeline/main.py --re-embed
```

### 7. Run Agentic RAG API

```bash
uv run -m agentic_rag.api
```

## Docker Setup and Usage Guide

### 📋 Prerequisites

Install Docker and Docker Compose:
```bash
# On macOS using Homebrew
brew install docker docker-compose

# On Ubuntu/Debian
sudo apt update && sudo apt install docker.io docker-compose

# On Windows - Download Docker Desktop
```

Verify Installation:
```bash
docker --version
docker-compose --version
```

### 🚀 Step-by-Step Docker Setup

#### Step 1: Prepare Environment

Copy environment file:
```bash
cp .env.docker .env
```

Edit `.env` file with your API keys:
```bash
nano .env  # or use your preferred editor
```

**Required API Keys:**
- `GEMINI_API_KEY`: Get from Google AI Studio
- `GROQ_API_KEY`: Get from Groq Console
- `OPENAI_API_KEY`: Get from OpenAI Platform
- `SERPER_API_KEY`: Get from Serper.dev

#### Step 2: Create Required Directories
```bash
# Create directories that will be mounted as volumes
mkdir -p resources artifacts docker config
```

#### Step 3: Build and Start Services
```bash
# Build and start all services in detached mode
docker-compose up -d --build
```

**What happens during this step:**
- **PostgreSQL**: Starts with PGVector extension enabled
- **Ollama**: Starts local LLM inference service
- **RAG App**: Builds your application image and starts the API server
- **Phoenix**: Starts observability dashboard (optional)

#### Step 4: Verify Services are Running
```bash
# Check all containers are healthy
docker-compose ps

# Check logs
docker-compose logs rag_app
docker-compose logs postgres
docker-compose logs ollama
```

#### Step 5: Download Ollama Models
```bash
# Enter Ollama container
docker-compose exec ollama bash

# Download required models
#ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull gemma3:1b

# Exit container
exit
```

#### Step 6: Process Your Documents

Add your documents to the resources folder:
```bash
cp your_documents.pdf resources/
```

Run the RAG pipeline:
```bash
# Process documents and create embeddings
docker-compose exec rag_app uv run rag_pipeline/main.py

# Or re-embed existing data
docker-compose exec rag_app uv run rag_pipeline/main.py --re-embed
```

### 🎯 Using the Docker Application

#### API Endpoints
Once running, your API will be available at:
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Phoenix Dashboard**: http://localhost:6006 (for observability)

#### Available Endpoints

**Basic RAG**: POST `/rag`
```bash
curl -X POST "http://localhost:8000/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the procurement standards?"}'
```

**Agentic RAG with CrewAI**: POST `/agentic_rag`
```bash
curl -X POST "http://localhost:8000/agentic_rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "How should vendors be evaluated?"}'
```

**Get Relevant Chunks**: POST `/relevant_chunks`
```bash
curl -X POST "http://localhost:8000/relevant_chunks" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the approval process?"}'
```

## Evaluation Framework

The system includes comprehensive evaluation capabilities using both RAGAS and Phoenix frameworks to assess RAG performance across multiple dimensions.

### 📊 RAGAS Evaluation

**RAGAS** (Retrieval Augmented Generation Assessment) provides automated evaluation of RAG systems using LLM-based metrics.

#### Key Metrics Evaluated:
- **Faithfulness**: Measures how grounded the generated answer is in the retrieved context
- **Answer Relevancy**: Evaluates how relevant the generated answer is to the given question
- **Context Precision**: Assesses the precision of retrieved context chunks
- **Context Recall**: Measures the coverage of relevant information in retrieved context

#### Running RAGAS Evaluation:
```bash
# Run RAGAS evaluation with predefined test queries
uv run evaluation/ragas_evaluation.py

# With custom CSV file containing RAG results
uv run evaluation/ragas_evaluation.py --rag-results artifacts/rag_results.csv
```

#### RAGAS Configuration:
```python
# Located in: evaluation/ragas_evaluation.py
TEST_QUERIES = [
    {
        "question": "What are the procurement standards?",
        "ground_truth": "Abu Dhabi procurement standards include transparency, competitiveness, and value for money."
    },
    # ... more test queries
]
```

#### Output:
- **CSV Results**: `artifacts/ragas_evaluation_results.csv`
- **Console Summary**: Metric averages and performance insights
- **Best/Worst Queries**: Identification of top and bottom performing queries

### 🔍 Phoenix Evaluation

**Phoenix** provides observability and evaluation capabilities with real-time tracing and comprehensive RAG assessment.

#### Key Features:
- **Live Tracing**: Real-time observation of RAG pipeline execution
- **Hallucination Detection**: Identifies potential hallucinations in generated responses
- **QA Correctness**: Evaluates the correctness of question-answering
- **Relevance Assessment**: Measures relevance of retrieved documents
- **Interactive Dashboard**: Web-based interface for evaluation exploration

#### Running Phoenix Evaluation:
```bash
# Start Phoenix evaluation with dashboard
uv run evaluation/phoenix_evaluation.py
```

#### Phoenix Dashboard:
Once running, access the Phoenix dashboard at:
- **Dashboard URL**: http://localhost:6006
- **Trace Exploration**: Interactive trace analysis
- **Metric Visualization**: Charts and graphs of evaluation metrics
- **Document Analysis**: Detailed context and retrieval analysis

#### Evaluation Outputs:
- **Queries DataFrame**: `artifacts/queries_df.csv`
- **Retrieved Documents**: `artifacts/retrieved_documents_df.csv`
- **Hallucination Evaluation**: `artifacts/hallucination_eval_df.csv`
- **QA Evaluation**: `artifacts/qa_eval_df.csv`
- **Relevance Evaluation**: `artifacts/relevance_eval_df.csv`

### 📈 Evaluation Workflow

#### 1. Setup Evaluation Environment
```bash
# Ensure artifacts directory exists
mkdir -p artifacts

# Set required API keys for evaluation models
export OPENAI_API_KEY="your-openai-key"  # Required for both RAGAS and Phoenix
```

#### 2. Prepare Test Dataset
```python
# Define your evaluation queries
test_queries = [
    {
        "question": "Your test question",
        "ground_truth": "Expected answer (optional)"
    }
]
```

#### 3. Run Comprehensive Evaluation
```bash
# Run both evaluation frameworks
uv run evaluation/ragas_evaluation.py
uv run evaluation/phoenix_evaluation.py

# Or run evaluation on existing RAG results
uv run evaluation/ragas_evaluation.py --rag-results artifacts/previous_results.csv
```

#### 4. Analyze Results
```bash
# View RAGAS results
cat artifacts/ragas_evaluation_results.csv

# Open Phoenix dashboard for interactive analysis
# Navigate to http://localhost:6006
```

### 🎯 Evaluation Best Practices

#### Test Query Design:
- **Diverse Topics**: Cover all major document themes
- **Varying Complexity**: Include simple facts and complex reasoning questions
- **Edge Cases**: Test boundary conditions and potential failure modes
- **Ground Truth**: Provide expected answers when possible for accurate assessment

#### Continuous Evaluation:
```bash
# Set up automated evaluation pipeline
# Run evaluation after each model or configuration change
uv run evaluation/ragas_evaluation.py
uv run evaluation/phoenix_evaluation.py

# Compare results across different configurations
diff artifacts/ragas_evaluation_results_v1.csv artifacts/ragas_evaluation_results_v2.csv
```

#### Performance Monitoring:
- **Baseline Establishment**: Run initial evaluation to establish performance baselines
- **Regular Assessment**: Periodic evaluation to detect performance drift
- **A/B Testing**: Compare different configurations using evaluation metrics
- **Error Analysis**: Deep dive into low-scoring queries for system improvement

## About Contextual RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) system that combines multiple advanced techniques for document processing, semantic search, and intelligent response generation. This implementation features contextual embedding enhancement, multi-model fallback systems, and agentic workflows for robust document Q&A capabilities.

## Implementation Approaches

### 1. Document Processing
**Approach Used: Docling-based Multi-format Document Conversion**

The system employs Docling (`langchain_docling`) as the primary document processing engine, which provides robust conversion capabilities for various document formats into structured markdown.

**Key Features:**
- **Universal Format Support**: Handles PDFs, Word documents, and other common formats
- **Structured Output**: Converts documents to clean markdown format preserving hierarchical structure
- **Metadata Preservation**: Maintains document metadata throughout the conversion process
- **Error Handling**: Robust error handling for corrupted or complex documents

**Implementation Details:**
```python
# Located in: rag_pipeline/data_extraction.py
loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
docs = loader.load()
```

### 2. Docling Data Pipeline & Storage
**Approach Used: Incremental Processing with Hash-based Change Detection**

The pipeline implements an efficient incremental processing system that only processes new or modified documents, significantly reducing processing time and computational overhead.

**Key Features:**
- **File Hash Monitoring**: SHA256-based file change detection system
- **Incremental Processing**: Only processes new or modified files
- **Artifact Management**: Organized storage of processed markdown files
- **Progress Tracking**: Resumable processing with progress persistence
- **Graceful Interruption**: Signal handling for safe pipeline interruption

**Implementation Details:**
```python
# Located in: rag_pipeline/file_hash_manager.py
class FileHashManager:
    def get_changed_or_new_files(self, resources_folder: str) -> Set[str]:
        # Returns only files that have changed or are new
```

### 3. LlamaIndex + PGVector/PostgreSQL RAG Methodology
**Approach Used: Production-grade Vector Storage with LlamaIndex Integration**

The system leverages LlamaIndex's robust framework combined with PostgreSQL's PGVector extension for scalable, persistent vector storage and retrieval.

**Key Features:**
- **Scalable Vector Storage**: PostgreSQL with PGVector extension for production-grade performance
- **LlamaIndex Integration**: Seamless document indexing and querying capabilities
- **Configurable Similarity Search**: HNSW indexing for efficient similarity searches
- **Persistent Storage**: Durable storage that survives application restarts
- **Metadata Filtering**: Rich metadata support for advanced filtering and retrieval

**Implementation Details:**
```python
# Located in: rag_pipeline/embedding_storage.py
vector_store = PGVectorStore.from_params(
    database=url.database,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=db_config['table_name'],
    embed_dim=embedding_config.get('dimension', 768),
    hnsw_kwargs=db_config.get('hnsw_kwargs', {}),
)
```

### 4. Contextual RAG (Anthropic-style) 
**Approach Used: Multi-stage Contextual Enhancement with Intelligent Chunking**

Inspired by Anthropic's contextual retrieval approach, this implementation enhances document chunks with contextual information to improve retrieval accuracy.

**Key Features:**
- **Semantic Chunking**: LLM-guided intelligent document segmentation that preserves topic boundaries
- **Contextual Enhancement**: Each chunk is enriched with context about its position within the larger document
- **Multi-document Context**: Handles large documents by processing them in segments and combining contexts
- **Fallback Resilience**: Multi-model fallback system ensures processing continues even if primary models fail

**Implementation Details:**
```python
# Located in: rag_pipeline/contextual_retrieval.py
def get_context_for_chunk(whole_document: str, chunk_content: str, config: ConfigManager):
    # Generates contextual information for each chunk
    # Uses prompt template to situate chunk within document context
```

**Semantic Chunking Process:**
```python
# Located in: rag_pipeline/chunking.py
def semantic_merge(text: str, config: ConfigManager) -> list[str]:
    # Uses LLM to detect topic boundaries and create semantically coherent chunks
```

### 5. Embedding / LLM / Re-ranking Models
**Approach Used: Multi-tier Model Architecture with Local and Cloud Integration**

The system implements a sophisticated multi-tier approach combining local and cloud-based models for optimal performance, cost-effectiveness, and reliability.

**Embedding Models:**
- **Primary**: Ollama-hosted `nomic-embed-text` (768-dimensional embeddings)
- **Local Processing**: All embeddings generated locally for privacy and speed

**LLM Models (Multi-tier Fallback):**
- **Primary**: Ollama-hosted models (Llama 3.1, Gemma 3)
- **Secondary**: Google Gemini (gemini-2.5-flash)
- **Tertiary**: Groq (llama-3.3-70b-versatile)
- **Quaternary**: AWS Bedrock (Claude)

**Re-ranking Implementation:**
```python
# Located in: agentic_rag/ollama_reranker.py
class OllamaReRanker:
    def rerank(self, query: str, documents: List[str], top_k: int = 5):
        # Uses local Ollama model to score and re-rank retrieved documents
```

**Key Features:**
- **Local-First Approach**: Primary processing on local models for privacy
- **Intelligent Fallback**: Automatic failover to cloud models when local models are unavailable
- **Re-ranking Enhancement**: Secondary relevance scoring to improve result quality
- **Cost Optimization**: Balances performance with API costs through strategic model selection

### 6. Locally Hosted Models via Ollama
**Approach Used: Self-hosted Model Infrastructure with Docker-like Simplicity**

The system leverages Ollama as the primary local model hosting solution, providing privacy, cost-effectiveness, and reliability for core operations.

**Key Features:**
- **Privacy-First**: All sensitive operations processed locally
- **Cost Effective**: No per-token charges for primary operations
- **Model Diversity**: Support for multiple model types (embedding, chat, specialized)
- **Easy Management**: Simple model switching and version control
- **Performance Optimization**: Local processing eliminates network latency

**Supported Local Models:**
- **Embedding**: `nomic-embed-text` - High-quality text embeddings
- **Chat**: `llama3.1:8b`, `gemma3` - General purpose conversation
- **Specialized**: Various models for specific tasks (chunking, re-ranking)

**Implementation Details:**
```python
# Located in: llm/llm.py
def generate_content(provider: str, model_name: str, prompt: str) -> str:
    if provider.lower() == "ollama":
        response = requests.post(OLLAMA_GENERATE_URL, json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=120)
```

### 7. Agentic Framework
**Approach Used: CrewAI-based Multi-Agent Orchestration**

The system implements an intelligent multi-agent framework using CrewAI that coordinates specialized agents for complex query handling and response synthesis.

**Agent Architecture:**
- **Retriever Agent**: Specializes in document search and knowledge retrieval
- **Response Synthesizer Agent**: Focuses on coherent response generation
- **Coordinator Logic**: Orchestrates agent interactions and task delegation

**Key Features:**
- **Specialized Roles**: Each agent has specific expertise and tools
- **Dynamic Tool Selection**: Agents can choose between RAG search and web search
- **Hierarchical Processing**: Sequential task execution with context passing
- **Fallback Mechanisms**: Web search when local knowledge is insufficient

**Implementation Details:**
```python
# Located in: agentic_rag/crew.py
@CrewBase
class AgenticRag:
    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retriever_agent'],
            tools=[self.rag_tool_instance, web_search_tool],
            llm=llm
        )
```

### 8. Prompt Optimization
**Approach Used: Template-based Prompt Engineering with Context-Aware Design**

The system employs carefully crafted prompt templates optimized for different stages of the RAG pipeline, ensuring consistent and high-quality outputs.

**Key Optimization Strategies:**
- **Context-Aware Prompts**: Different templates for different document sizes and complexities
- **Few-shot Learning**: Examples embedded in prompts for better model understanding
- **Role-based Prompting**: Clear role definitions for different agents and tasks
- **Output Formatting**: Structured prompts that ensure consistent output formats

**Contextual Retrieval Prompt:**
```python
prompt_template = """
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
```

**Agent-Specific Prompts:**
```yaml
# Located in: agentic_rag/config/agents.yaml
retriever_agent:
  role: >
    Retrieve relevant information to answer the user query
  goal: >
    Always try to use the rag search tool first. If unable to retrieve information, use web search tool
  backstory: >
    You're a meticulous analyst with a keen eye for detail
```

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Docling Pipeline│───▶│  Markdown Files │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Semantic Chunks │◀───│ Contextual RAG   │───▶│ Enhanced Chunks │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Embeddings    │◀───│ Ollama + PGVector│───▶│  Vector Storage │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Query Results  │◀───│ Multi-Agent RAG  │───▶│   Final Answer  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Features

- **🔄 Incremental Processing**: Only processes new or modified documents
- **🧠 Contextual Enhancement**: Anthropic-style contextual retrieval for improved accuracy  
- **🏠 Privacy-First**: Local model hosting with Ollama for sensitive operations
- **🎯 Intelligent Re-ranking**: Secondary relevance scoring with local models
- **🤖 Multi-Agent System**: CrewAI-powered agentic workflow for complex queries
- **📊 Production Ready**: PostgreSQL + PGVector for scalable vector storage
- **🛡️ Robust Fallbacks**: Multi-tier model architecture ensures high availability
- **📈 Resumable Processing**: Progress tracking allows interruption and continuation
- **🔍 Comprehensive Evaluation**: RAGAS and Phoenix frameworks for thorough assessment
- **🐳 Docker Support**: Containerized deployment for easy setup and scaling

This implementation combines cutting-edge RAG techniques with practical engineering considerations, resulting in a robust, scalable, and privacy-conscious document Q&A system with comprehensive evaluation capabilities.