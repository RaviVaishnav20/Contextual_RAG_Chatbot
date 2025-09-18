import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import time
import asyncio
import json
import requests
from typing import Optional, Dict, Any, List, Literal
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
import traceback
from sqlalchemy import create_engine, text, make_url

# OpenTelemetry imports for Phoenix tracing
import opentelemetry.trace as trace
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes

# Local imports
from agentic_rag.rag_rerank import get_rag_response, get_relevant_chunks_with_reranking
from agentic_rag.crew import AgenticRag
from config.config_manager import ConfigManager
from agentic_rag.eval.ragas import RAGASEvaluator

# Initialize tracing components (only when needed)
tracer_provider = None
tracer = None

def initialize_tracing():
    """Initialize tracing components lazily"""
    global tracer_provider, tracer
    if tracer_provider is None:
        try:
            tracer_provider = register(
                project_name="contextual_rag_chatbot",
                endpoint="http://phoenix:6006/v1/traces",
            )
            LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
            tracer = trace.get_tracer(__name__)
            return True
        except Exception as e:
            print(f"Failed to initialize tracing: {e}")
            return False
    return True

@contextmanager
def conditional_span(span_name: str, enable_tracing: bool = False, **attributes):
    """Context manager for conditional tracing"""
    if enable_tracing and initialize_tracing():
        with tracer.start_as_current_span(span_name) as span:
            # Set attributes if provided
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, str(value))
                except:
                    pass
            
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                raise
    else:
        # No-op span for consistency
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def get_span_context(self): 
                class NoOpContext:
                    trace_id = 0
                return NoOpContext()
        
        yield NoOpSpan()

# Initialize FastAPI app
app = FastAPI(
    title="Contextual RAG ChatBot API",
    description="Enhanced RAG API with conditional Phoenix tracing and comprehensive health checks",
    version="2.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
config = ConfigManager()
ragas_evaluator = None

def get_ragas_evaluator():
    global ragas_evaluator
    if ragas_evaluator is None and os.getenv("OPENAI_API_KEY"):
        try:
            ragas_evaluator = RAGASEvaluator()
        except Exception as e:
            print(f"Failed to initialize RAGAS evaluator: {e}")
    return ragas_evaluator

# Request/Response models
class Query(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    evaluate: bool = Field(default=False, description="Whether to run RAGAS evaluation and tracing")
    reference_answer: Optional[str] = Field(default="", description="Reference answer for evaluation")

class RAGResponse(BaseModel):
    retrieved_text: str
    llm_response: str
    response_time: float
    session_id: str
    query_id: str
    evaluation: Optional[Dict[str, Any]] = None
    phoenix_trace_id: Optional[str] = None

class AgenticResponse(BaseModel):
    context: str
    response: str
    response_time: float
    session_id: str
    query_id: str
    evaluation: Optional[Dict[str, Any]] = None
    phoenix_trace_id: Optional[str] = None

class ChunksResponse(BaseModel):
    retrieved_text: str
    response_time: float
    session_id: str
    query_id: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default="contextual-rag")
    messages: List[ChatMessage]

class HealthStatus(BaseModel):
    service: str
    status: str
    details: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None
    error: Optional[str] = None

# Utility functions
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

async def run_ragas_evaluation(query: str, response: str, contexts: List[str], reference: str = ""):
    """Run RAGAS evaluation in background"""
    evaluator = get_ragas_evaluator()
    if evaluator:
        return await evaluator.evaluate_response(query, response, contexts, reference)
    return None

def get_trace_id(span) -> Optional[str]:
    """Extract trace ID from span"""
    try:
        span_context = span.get_span_context()
        return f"{span_context.trace_id:032x}" if span_context.trace_id != 0 else None
    except:
        return None

# Health Check Functions
async def check_ollama_health() -> HealthStatus:
    """Check Ollama service health"""
    start_time = time.time()
    try:
        ollama_host = config.get_ollama_host()
        
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
        db_host = os.getenv("DATABASE_HOST", "localhost")
        db_port = os.getenv("DATABASE_PORT", "5433")
        db_name = os.getenv("DATABASE_NAME", "vector_db")
        db_user = os.getenv("DATABASE_USER", "ravi")
        db_password = os.getenv("DATABASE_PASSWORD", "password")
        
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
            table_name = os.getenv("DATABASE_TABLE_NAME", "contextual_embedding")
            table_check = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
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
                    "table_name": table_name
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
        response = requests.get("http://phoenix:6006/health", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return HealthStatus(
                service="phoenix",
                status="healthy",
                details={"endpoint": "http://phoenix:6006"},
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
        result = await get_relevant_chunks_with_reranking(test_query, initial_k=3, final_k=2)
        response_time = time.time() - start_time
        
        if result and "retrieved_text" in result:
            return HealthStatus(
                service="rag_pipeline",
                status="healthy",
                details={
                    "test_query": test_query,
                    "chunks_retrieved": len(extract_contexts_from_retrieved_text(result["retrieved_text"]))
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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Contextual RAG ChatBot API is running",
        "status": "healthy",
        "version": "2.2.0",
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed",
            "ollama_health": "/health/ollama",
            "postgres_health": "/health/postgres",
            "phoenix_health": "/health/phoenix",
            "rag_health": "/health/rag",
            "rag_endpoint": "/rag",
            "agentic_rag": "/agentic_rag",
            "openai_compatible": "/v1/chat/completions"
        }
    }

@app.get("/health")
async def health_check():
    """Quick health check"""
    return {
        "api": "healthy",
        "tracing": "conditional",
        "evaluation": "available" if os.getenv("OPENAI_API_KEY") else "unavailable",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check for all services"""
    results = await asyncio.gather(
        check_ollama_health(),
        check_postgres_health(),
        check_phoenix_health(),
        check_rag_pipeline_health(),
        return_exceptions=True
    )
    
    health_results = []
    overall_status = "healthy"
    
    for result in results:
        if isinstance(result, Exception):
            health_results.append(HealthStatus(
                service="unknown",
                status="error",
                error=str(result)
            ))
            overall_status = "unhealthy"
        else:
            health_results.append(result)
            if result.status != "healthy":
                overall_status = "unhealthy"
    
    return {
        "overall_status": overall_status,
        "services": [result.dict() for result in health_results],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/ollama")
async def ollama_health_check():
    """Dedicated Ollama health check"""
    return (await check_ollama_health()).dict()

@app.get("/health/postgres")
async def postgres_health_check():
    """Dedicated PostgreSQL health check"""
    return (await check_postgres_health()).dict()

@app.get("/health/phoenix")
async def phoenix_health_check():
    """Dedicated Phoenix health check"""
    return (await check_phoenix_health()).dict()

@app.get("/health/rag")
async def rag_health_check():
    """Dedicated RAG pipeline health check"""
    return (await check_rag_pipeline_health()).dict()

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(query: Query, background_tasks: BackgroundTasks):
    """Enhanced RAG endpoint with conditional tracing and evaluation"""
    session_id, query_id = generate_ids()
    start_time = time.time()
    
    # Use conditional tracing based on evaluate flag
    with conditional_span(
        "rag_query", 
        enable_tracing=query.evaluate,
        **{
            SpanAttributes.INPUT_VALUE: query.query,
            "session_id": session_id,
            "query_id": query_id,
            "user_id": query.user_id or "anonymous"
        }
    ) as span:
        
        try:
            # Get RAG response with timeout
            response = await asyncio.wait_for(
                get_rag_response(query.query), 
                timeout=120  # 2 minute timeout
            )
            response_time = time.time() - start_time
            
            # Set output attributes if tracing
            if query.evaluate:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, response["llm_response"])
                span.set_attribute("response_time", response_time)
            
            # Get trace ID
            trace_id = get_trace_id(span) if query.evaluate else None
            
            # Handle evaluation
            evaluation_result = None
            if query.evaluate and os.getenv("OPENAI_API_KEY"):
                contexts = extract_contexts_from_retrieved_text(response["retrieved_text"])
                
                background_tasks.add_task(
                    run_ragas_evaluation, 
                    query.query, 
                    response["llm_response"], 
                    contexts, 
                    query.reference_answer
                )
                
                evaluation_result = {
                    "status": "evaluating", 
                    "message": "RAGAS evaluation running in background"
                }
            
            return RAGResponse(
                retrieved_text=response["retrieved_text"],
                llm_response=response["llm_response"],
                response_time=response_time,
                session_id=session_id,
                query_id=query_id,
                evaluation=evaluation_result,
                phoenix_trace_id=trace_id
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="RAG request timed out after 120 seconds")
        except Exception as e:
            error_details = f"RAG processing failed: {str(e)}"
            print(f"RAG Error: {error_details}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_details)

@app.post("/agentic_rag", response_model=AgenticResponse)
async def agentic_rag_endpoint(query: Query, background_tasks: BackgroundTasks):
    """Enhanced Agentic RAG endpoint with conditional tracing and evaluation"""
    session_id, query_id = generate_ids()
    start_time = time.time()
    
    with conditional_span(
        "agentic_rag_query",
        enable_tracing=query.evaluate,
        **{
            SpanAttributes.INPUT_VALUE: query.query,
            "session_id": session_id,
            "query_id": query_id,
            "user_id": query.user_id or "anonymous"
        }
    ) as span:
        
        try:
            # Get Agentic RAG response with timeout
            agentic_rag = AgenticRag()
            
            # Conditional crew tracing
            if query.evaluate:
                with conditional_span("crew_kickoff", True, query=query.query):
                    response = await asyncio.wait_for(
                        asyncio.create_task(asyncio.to_thread(agentic_rag.run_crew_with_context, query.query)),
                        timeout=180  # 3 minute timeout for agentic RAG
                    )
            else:
                response = await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(agentic_rag.run_crew_with_context, query.query)),
                    timeout=180
                )
                
            response_time = time.time() - start_time
            
            # Set output attributes if tracing
            if query.evaluate:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response["response"]))
                span.set_attribute("response_time", response_time)
                span.set_attribute("context_length_chars", len(response.get("context", "")))
            
            # Get trace ID
            trace_id = get_trace_id(span) if query.evaluate else None
            
            # Handle evaluation
            evaluation_result = None
            if query.evaluate and os.getenv("OPENAI_API_KEY"):
                contexts = extract_contexts_from_retrieved_text(response["context"])
                
                background_tasks.add_task(
                    run_ragas_evaluation, 
                    query.query, 
                    str(response["response"]), 
                    contexts, 
                    query.reference_answer
                )
                
                evaluation_result = {
                    "status": "evaluating", 
                    "message": "RAGAS evaluation running in background"
                }
            
            return AgenticResponse(
                context=response["context"],
                response=str(response["response"]),
                response_time=response_time,
                session_id=session_id,
                query_id=query_id,
                evaluation=evaluation_result,
                phoenix_trace_id=trace_id
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Agentic RAG request timed out after 180 seconds")
        except Exception as e:
            error_details = f"Agentic RAG processing failed: {str(e)}"
            print(f"Agentic RAG Error: {error_details}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_details)

@app.post("/relevant_chunks", response_model=ChunksResponse)
async def relevant_chunks_endpoint(query: Query):
    """Retrieve relevant chunks with conditional tracing"""
    session_id, query_id = generate_ids()
    start_time = time.time()
    
    with conditional_span(
        "retrieve_chunks",
        enable_tracing=query.evaluate,
        **{
            SpanAttributes.INPUT_VALUE: query.query,
            "session_id": session_id,
            "query_id": query_id
        }
    ) as span:
        
        try:
            response = await asyncio.wait_for(
                get_relevant_chunks_with_reranking(query.query),
                timeout=60  # 1 minute timeout
            )
            response_time = time.time() - start_time
            
            if query.evaluate:
                span.set_attribute("response_time", response_time)
                span.set_attribute("chunks_retrieved", len(extract_contexts_from_retrieved_text(response["retrieved_text"])))
            
            return ChunksResponse(
                retrieved_text=response["retrieved_text"],
                response_time=response_time,
                session_id=session_id,
                query_id=query_id
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Chunk retrieval timed out after 60 seconds")
        except Exception as e:
            error_details = f"Chunk retrieval failed: {str(e)}"
            print(f"Chunk Retrieval Error: {error_details}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_details)

# OpenAI-compatible endpoints
@app.get("/v1/models")
async def list_models():
    """List models for OpenAI compatibility"""
    now = int(time.time())
    return {
        "object": "list",
        "data": [{
            "id": "contextual-rag",
            "object": "model",
            "created": now,
            "owned_by": "local"
        }]
    }

@app.post("/v1/chat/completions")
async def openai_compatible_endpoint(request: ChatCompletionRequest):
    """OpenAI-compatible endpoint for Open WebUI integration"""
    try:
        messages = request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last user message
        user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query_text = user_message.content
        requested_model = request.model or "contextual-rag"
        
        # Use agentic RAG without evaluation/tracing for OpenAI compatibility
        query = Query(query=query_text, evaluate=False)
        response = await agentic_rag_endpoint(query, BackgroundTasks())
        
        # Return in OpenAI format
        return {
            "id": f"chatcmpl-{response.query_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(query_text.split()),
                "completion_tokens": len(response.response.split()),
                "total_tokens": len(query_text.split()) + len(response.response.split())
            }
        }
        
    except Exception as e:
        error_details = f"Chat completion failed: {str(e)}"
        print(f"OpenAI Compatible Error: {error_details}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_details)

# Evaluation endpoints
@app.post("/evaluate/batch")
async def batch_evaluate_endpoint(queries: List[Dict[str, str]]):
    """Batch RAGAS evaluation endpoint with tracing"""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key not configured for evaluation")
    
    # Initialize tracing for batch evaluation
    initialize_tracing()
    
    with tracer.start_as_current_span("batch_evaluation") as span:
        span.set_attribute("batch_size", len(queries))
        
        try:
            results = []
            for i, query_data in enumerate(queries):
                query_text = query_data.get("query", "")
                reference = query_data.get("reference", "")
                
                with tracer.start_as_current_span(f"evaluate_query_{i}") as query_span:
                    query_span.set_attribute(SpanAttributes.INPUT_VALUE, query_text)
                    
                    # Get RAG response
                    response = await get_rag_response(query_text)
                    contexts = extract_contexts_from_retrieved_text(response["retrieved_text"])
                    
                    # Evaluate
                    evaluation = await run_ragas_evaluation(query_text, response["llm_response"], contexts, reference)
                    
                    query_span.set_attribute("evaluation_score", str(evaluation))
                    
                    results.append({
                        "query": query_text,
                        "response": response["llm_response"],
                        "contexts": contexts,
                        "evaluation": evaluation
                    })
            
            # Save batch results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"artifacts/batch_evaluation_{timestamp}.csv", index=False)
            
            span.set_attribute("results_saved", f"artifacts/batch_evaluation_{timestamp}.csv")
            
            return {"message": "Batch evaluation completed", "results": results}
            
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error_message", str(e))
            raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)