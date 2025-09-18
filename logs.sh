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
