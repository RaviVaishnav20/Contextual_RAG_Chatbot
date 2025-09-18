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
