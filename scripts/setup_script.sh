#!/bin/bash

# Setup script for Contextual RAG ChatBot with Open WebUI, Phoenix, and RAGAS
echo "🚀 Setting up Contextual RAG ChatBot with Open WebUI Integration"
echo "================================================================="

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p openwebui/functions
mkdir -p artifacts/evaluations
mkdir -p config/phoenix
mkdir -p docker

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here

# AWS Credentials (optional, for Bedrock)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Database Configuration
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=vector_db
DATABASE_USER=ravi
DATABASE_PASSWORD=password

# Open WebUI Configuration
WEBUI_SECRET_KEY=$(openssl rand -hex 32)

# Phoenix Configuration
PHOENIX_SQL_DATABASE_URL=postgresql://ravi:password@postgres:5432/vector_db
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006

# RAGAS Configuration
ENABLE_RAGAS_EVALUATION=true
EOF
    echo "✅ Created .env template. Please fill in your API keys."
fi

# Copy Open WebUI function
echo "📋 Setting up Open WebUI functions..."
cat > openwebui/functions/contextual_rag.py << 'EOF'
"""
Contextual RAG Function for Open WebUI
Integrates with the Contextual RAG ChatBot backend
"""

import requests
import json
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Function:
    """
    Contextual RAG integration for Open WebUI
    Provides intelligent document search and retrieval
    """
    
    class Valves(BaseModel):
        rag_api_url: str = "http://rag_app:8000"
        default_mode: str = "agentic"  # "basic" or "agentic"
        enable_evaluation: bool = True
        auto_search_threshold: float = 0.7  # Threshold for automatic RAG search
        
    def __init__(self):
        self.valves = self.Valves()
    
    async def pipe(
        self, body: dict, __user__: dict, __event_emitter__=None, __task__: str = None
    ) -> str:
        """
        Process user messages and enhance with RAG when appropriate
        """
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Get the last user message
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return body
            
        query = last_message.get("content", "")
        
        # Determine if we should use RAG
        should_use_rag = self._should_use_rag(query)
        
        if should_use_rag:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Searching knowledge base...", "done": False}
                })
            
            try:
                # Perform RAG search
                rag_result = await self._perform_rag_search(query, __user__.get("id"))
                
                if rag_result.get("status") == "success":
                    # Enhance the message with RAG context
                    enhanced_content = self._format_rag_response(query, rag_result)
                    last_message["content"] = enhanced_content
                    
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status", 
                            "data": {"description": "Knowledge retrieved successfully", "done": True}
                        })
                else:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Knowledge search failed", "done": True}
                        })
                        
            except Exception as e:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"RAG error: {str(e)}", "done": True}
                    })
        
        return body
    
    def _should_use_rag(self, query: str) -> bool:
        """Determine if query should trigger RAG search"""
        rag_triggers = [
            "what is", "explain", "tell me about", "how to", "define",
            "describe", "find information", "search for", "look up",
            "procedure", "process", "standard", "policy", "guideline"
        ]
        
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in rag_triggers)
    
    async def _perform_rag_search(self, query: str, user_id: str) -> Dict[str, Any]:
        """Perform RAG search using the backend API"""
        try:
            endpoint = "/agentic_rag" if self.valves.default_mode == "agentic" else "/rag"
            
            payload = {
                "query": query,
                "user_id": user_id,
                "evaluate": self.valves.enable_evaluation
            }
            
            response = requests.post(
                f"{self.valves.rag_api_url}{endpoint}",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Search timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _format_rag_response(self, original_query: str, rag_result: Dict[str, Any]) -> str:
        """Format RAG response for display"""
        answer = rag_result.get("response", rag_result.get("llm_response", ""))
        response_time = rag_result.get("response_time", 0)
        mode = rag_result.get("mode", self.valves.default_mode)
        
        formatted = f"""Based on your question: "{original_query}"

{answer}

---
📊 Search completed in {response_time:.2f}s using {mode} mode
🔍 Phoenix Trace: {rag_result.get('phoenix_trace_id', 'N/A')}"""

        if rag_result.get("evaluation"):
            eval_info = rag_result["evaluation"]
            if eval_info.get("status") == "evaluating":
                formatted += "\n📈 Quality evaluation in progress..."
        
        return formatted
EOF

echo "✅ Open WebUI function created"

# Create Phoenix configuration
echo "⚡ Setting up Phoenix configuration..."
cat > config/phoenix/phoenix.yml << EOF
# Phoenix Configuration
project_name: "contextual_rag_chatbot"
database_url: "${PHOENIX_SQL_DATABASE_URL}"
collector_endpoint: "${PHOENIX_COLLECTOR_ENDPOINT}"

# Trace configuration
tracing:
  enabled: true
  sample_rate: 1.0
  export_timeout: 30000

# Evaluation configuration
evaluation:
  enabled: true
  auto_evaluate: false
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
EOF

# Create startup script
echo "🚀 Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Contextual RAG ChatBot with Open WebUI"
echo "=================================================="

# Check if .env file exists and has required keys
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one using the template."
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

# Check for required API keys
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set. Some features may not work."
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. RAGAS evaluation will be disabled."
fi

echo "📦 Pulling latest images..."
docker-compose pull

echo "🏗️  Building containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check Postgres
if docker-compose exec -T postgres pg_isready -U ravi -d vector_db > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Ollama
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "✅ Ollama is ready"
else
    echo "❌ Ollama is not ready"
fi

# Check RAG App
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ RAG App is ready"
else
    echo "❌ RAG App is not ready"
fi

# Check Phoenix
if curl -s http://localhost:6006/health > /dev/null 2>&1; then
    echo "✅ Phoenix is ready"
else
    echo "❌ Phoenix is not ready"
fi

# Check Open WebUI
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Open WebUI is ready"
else
    echo "❌ Open WebUI is not ready"
fi

echo ""
echo "🎉 Setup complete! Access your services:"
echo "├── 🤖 Open WebUI: http://localhost:3000"
echo "├── 📊 Phoenix Observability: http://localhost:6006"
echo "├── 🔧 RAG API: http://localhost:8000/docs"
echo "└── 📡 Ollama: http://localhost:11434"
echo ""
echo "💡 Tips:"
echo "• First time? Run 'docker-compose exec rag_app python -m rag_pipeline.main' to process documents"
echo "• View logs: docker-compose logs -f [service_name]"
echo "• Stop services: docker-compose down"
EOF

chmod +x start.sh

# Create requirements update script
echo "📋 Creating requirements update script..."
cat > update_requirements.sh << 'EOF'
#!/bin/bash

echo "📦 Updating Python requirements for Phoenix and RAGAS integration..."

# Add to pyproject.toml
cat >> pyproject.toml << 'PYPROJECT'

# Phoenix and Observability
phoenix-ai = "^4.0.0"
phoenix-evals = "^0.13.0"
openinference-instrumentation-llama-index = "^2.0.0"
opentelemetry-api = "^1.21.0"
opentelemetry-sdk = "^1.21.0"
opentelemetry-exporter-otlp = "^1.21.0"

# RAGAS for evaluation
ragas = "^0.1.7"
langchain-openai = "^0.1.0"

# Additional async support
aiohttp = "^3.9.0"
asyncio-timeout = "^4.0.0"

# Enhanced FastAPI features
fastapi = {extras = ["all"], version = "^0.104.0"}
uvicorn = {extras = ["standard"], version = "^0.24.0"}
PYPROJECT

echo "✅ Requirements updated. Run 'uv sync' to install new dependencies."
EOF

chmod +x update_requirements.sh

echo "🔧 Creating maintenance scripts..."

# Create logs script
cat > logs.sh << 'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "📋 Available services: postgres, ollama, rag_app, open-webui, phoenix"
    echo "Usage: ./logs.sh [service_name]"
    echo "Or: ./logs.sh all (for all services)"
else
    if [ "$1" = "all" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f $1
    fi
fi
EOF

chmod +x logs.sh

# Create stop script
cat > stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping all services..."
docker-compose down
echo "✅ All services stopped"
EOF

chmod +x stop.sh

# Create evaluation script
cat > run_evaluation.sh << 'EOF'
#!/bin/bash
echo "📊 Running RAGAS evaluation..."

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Error: OPENAI_API_KEY must be set for evaluation"
    exit 1
fi

docker-compose exec rag_app python -m evaluation.ragas_evaluation
echo "✅ Evaluation complete. Check artifacts/ragas_evaluation_results.csv"
EOF

chmod +x run_evaluation.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 What was created:"
echo "├── 🐳 Updated docker-compose.yaml with Open WebUI and Phoenix"
echo "├── 🔧 Enhanced API with Phoenix tracing and RAGAS evaluation"
echo "├── 🌐 Open WebUI custom functions for RAG integration"
echo "├── 📊 Phoenix observability configuration"
echo "├── 🚀 start.sh - Main startup script"
echo "├── 📦 update_requirements.sh - Update Python dependencies"
echo "├── 📋 logs.sh - View service logs"
echo "├── 🛑 stop.sh - Stop all services"
echo "└── 📊 run_evaluation.sh - Run RAGAS evaluation"
echo ""
echo "🚀 Next Steps:"
echo "1. Fill in your API keys in .env file"
echo "2. Run: ./update_requirements.sh"
echo "3. Run: ./start.sh"
echo "4. Visit http://localhost:3000 for Open WebUI"
echo "5. Visit http://localhost:6006 for Phoenix observability"
echo ""
echo "💡 For first-time setup, also run:"
echo "   docker-compose exec rag_app python -m rag_pipeline.main"
echo "   to process your documents and create embeddings."