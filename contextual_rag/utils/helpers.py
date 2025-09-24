import os
from dotenv import load_dotenv
import time
import requests
import uuid
from sqlalchemy import create_engine, text
# Local imports
from contextual_rag.application.rag.rag import get_rag_answer
from contextual_rag.model.inference.api.models import HealthStatus
from contextual_rag.model.evaluation.ragas import RagasEvaluator
from contextual_rag.infrastructure.config_manager import ConfigManager
from typing import Optional, List
# Load variables from .env into environment
load_dotenv()
cm = ConfigManager()
ragas_evaluator = None

# Health Check Functions
async def check_ollama_health() -> HealthStatus:
    """Check Ollama service health"""
    start_time = time.time()
    try:
        ollama_host = cm.get_ollama_host()
        
        # Check if Ollama is responding
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            return HealthStatus(
                service="ollama",
                status="healthy",
                details={
                    "host": ollama_host,
                    "available_models": len(models),
                    "models": [model.get("name", "unknown") for model in models[:5]]  # First 5 models
                },
                response_time=response_time
            )
        else:
            return HealthStatus(
                service="ollama",
                status="unhealthy",
                error=f"HTTP {response.status_code}",
                response_time=response_time
            )
            
    except Exception as e:
        return HealthStatus(
            service="ollama",
            status="error",
            error=str(e),
            response_time=time.time() - start_time
        )

async def check_postgres_health() -> HealthStatus:
    """Check PostgreSQL database health"""
    start_time = time.time()
    try:
        db_cfg = cm.get_database_config() or {}
        
        db_host = db_cfg.get('host', 'localhost')
        db_port = db_cfg.get('port', '5432')
        db_name = db_cfg.get('database', 'vector_db')
        db_table_name = db_cfg.get('table_name', 'test_embed')
        db_user = db_cfg.get('user', 'ravi')
        db_password = db_cfg.get('password', 'password')
        db_hnsw_kwargs = db_cfg.get('hnsw_kwargs', {})

        
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        engine = create_engine(connection_string, pool_timeout=10)
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1 as test"))
            test_result = result.fetchone()
            
            # Check if vector extension is available
            vector_check = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
            has_vector = vector_check.fetchone()[0]
            
            # Check if our table exists
            
            table_check = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{db_table_name}'
                )
            """))
            has_table = table_check.fetchone()[0]
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service="postgres",
                status="healthy",
                details={
                    "host": f"{db_host}:{db_port}",
                    "database": db_name,
                    "vector_extension": has_vector,
                    "embedding_table_exists": has_table,
                    "table_name": db_table_name
                },
                response_time=response_time
            )
            
    except Exception as e:
        return HealthStatus(
            service="postgres",
            status="error",
            error=str(e),
            response_time=time.time() - start_time
        )

async def check_phoenix_health() -> HealthStatus:
    """Check Phoenix tracing service health"""
    start_time = time.time()
    try:
        phoenix_cfg = cm.get_phoenix_config() or {}
        endpoint = phoenix_cfg.get('tracing', {}).get('endpoint', 'http://localhost:6006')
        health_endpoint = phoenix_cfg.get('tracing', {}).get('health_endpoint', 'http://localhost:6006/health')
        response = requests.get(endpoint, timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return HealthStatus(
                service="phoenix",
                status="healthy",
                details={"endpoint": endpoint},
                response_time=response_time
            )
        else:
            return HealthStatus(
                service="phoenix",
                status="unhealthy",
                error=f"HTTP {response.status_code}",
                response_time=response_time
            )
            
    except Exception as e:
        return HealthStatus(
            service="phoenix",
            status="error",
            error=str(e),
            response_time=time.time() - start_time
        )

async def check_rag_pipeline_health() -> HealthStatus:
    """Test the RAG pipeline with a simple query"""
    start_time = time.time()
    try:
        # Test with a simple query
        test_query = "test"
        result = await get_rag_answer(test_query)
        response_time = time.time() - start_time
        
        if len(result.answer) > 0:
            return HealthStatus(
                service="rag_pipeline",
                status="healthy",
                details={
                    "test_query": test_query,
                    "answer": result[0]
                },
                response_time=response_time
            )
        else:
            return HealthStatus(
                service="rag_pipeline",
                status="unhealthy",
                error="No results returned",
                response_time=response_time
            )
            
    except Exception as e:
        return HealthStatus(
            service="rag_pipeline",
            status="error",
            error=str(e),
            response_time=time.time() - start_time
        )

def get_ragas_evaluator():
    global ragas_evaluator
    if ragas_evaluator is None and os.getenv("OPENAI_API_KEY"):
        try:
            ragas_evaluator = RagasEvaluator()
        except Exception as e:
            print(f"Failed to initialize RAGAS evaluator: {e}")
    return ragas_evaluator

def generate_ids():
    """Generate session and query IDs"""
    return str(uuid.uuid4()), str(uuid.uuid4())

def extract_contexts_from_retrieved_text(retrieved_text: str) -> List[str]:
    """Extract individual contexts from retrieved text"""
    if not retrieved_text:
        return []
    
    chunks = retrieved_text.split("--- Chunk")
    contexts = []
    
    for chunk in chunks:
        if chunk.strip():
            lines = chunk.split('\n')
            content_lines = []
            skip_metadata = True
            
            for line in lines:
                if skip_metadata and ('Score:' in line or 'Source:' in line):
                    continue
                skip_metadata = False
                if line.strip():
                    content_lines.append(line.strip())
            
            if content_lines:
                contexts.append('\n'.join(content_lines))
    
    return contexts if contexts else [retrieved_text]

def run_ragas_evaluation(query: str, response: str, contexts: List[str], reference: str = ""):
    """Run RAGAS evaluation in background"""
    evaluator = get_ragas_evaluator()
    if evaluator:
        return evaluator.evaluate_rag(query, response, contexts, reference)
    return None

def get_trace_id(span) -> Optional[str]:
    """Extract trace ID from span"""
    try:
        span_context = span.get_span_context()
        return f"{span_context.trace_id:032x}" if span_context.trace_id != 0 else None
    except:
        return None

