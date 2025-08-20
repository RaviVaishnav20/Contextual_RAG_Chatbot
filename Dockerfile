# Use Python 3.13 slim image for smaller size
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for PostgreSQL, compilation, and other tools
RUN apt-get update && apt-get install -y \
    # PostgreSQL client and development headers
    libpq-dev \
    # Build tools for compiling Python packages
    build-essential \
    # Git for potential git operations
    git \
    # Curl for health checks and downloads
    curl \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies using uv
RUN uv sync 

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/markdown_artifacts

# Set Python path to include the application directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for FastAPI
EXPOSE 8000

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Default command to run the API server
CMD ["uv", "run", "-m", "agentic_rag.api"]

