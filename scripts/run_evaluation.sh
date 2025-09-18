#!/bin/bash
echo "📊 Running RAGAS evaluation..."

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Error: OPENAI_API_KEY must be set for evaluation"
    exit 1
fi

docker-compose exec rag_app python -m evaluation.ragas_evaluation
echo "✅ Evaluation complete. Check artifacts/ragas_evaluation_results.csv"
