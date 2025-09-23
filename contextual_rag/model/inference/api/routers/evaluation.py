# from fastapi import APIRouter, HTTPException
# from openwebui.models import Query
# from utils.tracing import initialize_tracing, tracer
# from agentic_rag.rag_rerank import get_rag_response
# from utils.helpers import extract_contexts_from_retrieved_text, run_ragas_evaluation
# from openinference.semconv.trace import SpanAttributes
# from datetime import datetime
# import os, pandas as pd
# from typing import Dict, List
# router = APIRouter()

# @router.post("/batch")
# async def batch_evaluate(queries: List[Dict[str, str]]):
#     """Batch RAGAS evaluation endpoint with tracing"""
#     if not os.getenv("OPENAI_API_KEY"):
#         raise HTTPException(status_code=400, detail="OpenAI API key not configured for evaluation")
    
#     # Initialize tracing for batch evaluation
#     initialize_tracing()
    
#     with tracer.start_as_current_span("batch_evaluation") as span:
#         span.set_attribute("batch_size", len(queries))
        
#         try:
#             results = []
#             for i, query_data in enumerate(queries):
#                 query_text = query_data.get("query", "")
#                 reference = query_data.get("reference", "")
                
#                 with tracer.start_as_current_span(f"evaluate_query_{i}") as query_span:
#                     query_span.set_attribute(SpanAttributes.INPUT_VALUE, query_text)
                    
#                     # Get RAG response
#                     response = await get_rag_response(query_text)
#                     contexts = extract_contexts_from_retrieved_text(response["retrieved_text"])
                    
#                     # Evaluate
#                     evaluation = await run_ragas_evaluation(query_text, response["llm_response"], contexts, reference)
                    
#                     query_span.set_attribute("evaluation_score", str(evaluation))
                    
#                     results.append({
#                         "query": query_text,
#                         "response": response["llm_response"],
#                         "contexts": contexts,
#                         "evaluation": evaluation
#                     })
            
#             # Save batch results
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             results_df = pd.DataFrame(results)
#             results_df.to_csv(f"artifacts/batch_evaluation_{timestamp}.csv", index=False)
            
#             span.set_attribute("results_saved", f"artifacts/batch_evaluation_{timestamp}.csv")
            
#             return {"message": "Batch evaluation completed", "results": results}
            
#         except Exception as e:
#             span.set_attribute("error", True)
#             span.set_attribute("error_message", str(e))
#             raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")
#     # keep your batch evaluation logic here...
