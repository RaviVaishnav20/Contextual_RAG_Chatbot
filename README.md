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
```
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

This implementation combines cutting-edge RAG techniques with practical engineering considerations, resulting in a robust, scalable, and privacy-conscious document Q&A system.