from fastapi import APIRouter, HTTPException, BackgroundTasks
from openwebui.models import Query, RAGResponse, AgenticResponse, ChunksResponse
from utils.helpers import generate_ids, extract_contexts_from_retrieved_text, run_ragas_evaluation, get_trace_id
from utils.tracing import conditional_span
from agentic_rag.rag_rerank import get_rag_response, get_relevant_chunks_with_reranking
from agentic_rag.crew import AgenticRag
import asyncio, time, traceback
from openinference.semconv.trace import SpanAttributes
import os

router = APIRouter()

@router.post("/rag", response_model=RAGResponse)
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

@router.post("/agentic_rag", response_model=AgenticResponse)
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

@router.post("/relevant_chunks", response_model=ChunksResponse)
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
