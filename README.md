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

