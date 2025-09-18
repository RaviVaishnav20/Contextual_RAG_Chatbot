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
