from fastapi import APIRouter, HTTPException, BackgroundTasks
from contextual_rag.model.inference.api.models import Query, RAGResponse, AgenticResponse, ChunksResponse
from contextual_rag.utils.helpers import generate_ids, run_ragas_evaluation, get_trace_id
from contextual_rag.utils.tracing import conditional_span
from contextual_rag.application.rag.rag import get_rag_answer, get_relevant_chunks_with_reranking
from contextual_rag.application.agents.crew.crew import AgenticRag
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
                get_rag_answer(query.query),
                timeout=120  # 2 minute timeout
            )
            response_time = time.time() - start_time
            
            # Set output attributes if tracing
            if query.evaluate:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, response[0])
                span.set_attribute("response_time", response_time)
            
            # Get trace ID
            trace_id = get_trace_id(span) if query.evaluate else None
            
            # Handle evaluation
            evaluation_result = None
            if query.evaluate and os.getenv("OPENAI_API_KEY"):
                contexts = response[1]
                
                background_tasks.add_task(
                    run_ragas_evaluation, 
                    query.query, 
                    response[0], 
                    contexts, 
                    query.reference_answer
                )
                
                evaluation_result = {
                    "status": "evaluating", 
                    "message": "RAGAS evaluation running in background"
                }
            
            return RAGResponse(
                retrieved_text=response[1],
                llm_response=response[0],
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
                        asyncio.create_task(asyncio.to_thread(agentic_rag.run_crew, query.query)),
                        timeout=180  # 3 minute timeout for agentic RAG
                    )
            else:
                response = await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(agentic_rag.run_crew, query.query)),
                    timeout=180
                )
                
            response_time = time.time() - start_time
            
            # Set output attributes if tracing
            if query.evaluate:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
                span.set_attribute("response_time", response_time)
            
            # Get trace ID
            trace_id = get_trace_id(span) if query.evaluate else None
            
            # Handle evaluation
            evaluation_result = None
            if query.evaluate and os.getenv("OPENAI_API_KEY"):
         
                
                background_tasks.add_task(
                    run_ragas_evaluation, 
                    query.query, 
                    str(response), 
                    [], 
                    query.reference_answer
                )
                
                evaluation_result = {
                    "status": "evaluating", 
                    "message": "RAGAS evaluation running in background"
                }
            
            return AgenticResponse(
                response=str(response),
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
                span.set_attribute("chunks_retrieved", len(response[1]))
            
            return ChunksResponse(
                retrieved_text=response,
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
