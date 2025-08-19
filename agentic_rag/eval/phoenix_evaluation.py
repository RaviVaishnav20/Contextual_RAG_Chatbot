import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import nest_asyncio
import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals
)
import opentelemetry.trace as trace
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations
import pandas as pd
from typing import List, Dict
import asyncio
from agentic_rag.rag_rerank import get_rag_response
from config.config_manager import ConfigManager
nest_asyncio.apply() 
# Initialize Phoenix and OpenAI
px.launch_app()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
# from phoenix.otel import register

# tracer_provider = register(
#   project_name="default",
#   endpoint="http://localhost:6006/v1/traces",
#   auto_instrument=True
# )



class PhoenixRAGEvaluator:
    """Simple Phoenix evaluation for your RAG system using GPT-4o-mini"""
    
    def __init__(self):
        self.config = ConfigManager()
        # Use GPT-4o-mini for cost-effective evaluation
        self.eval_model = px.evals.OpenAIModel(
            model="gpt-4.1-nano",
            temperature=0.0
        )
        # self.queries_df = get_qa_with_reference(px.Client())
        # self.retrieved_documents_df = get_retrieved_documents(px.Client())
    
    async def evaluate_queries(self, test_queries: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate RAG responses using Phoenix metrics
        
        Args:
            test_queries: List of dicts with 'query' and optional 'expected_answer'
        """
        print("🔍 Running Phoenix RAG Evaluation...")
        
        # Generate responses
        results = []
        for i, query_data in enumerate(test_queries):
            query = query_data['query']
            # expected = query_data.get('expected_answer', '')
            
            print(f"Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            try:
            # Get RAG response
                rag_result = await get_rag_response(query)
            # current_span = trace.get_current_span()
            # span_id = current_span.context.span_id if current_span else None
            # print("span_id")
            # print(span_id)
                
    #             # normalized = normalize_for_phoenix(rag_result)
                reference = str(rag_result["retrieved_text"])
                response = str(rag_result["llm_response"])

                results.append({"reference": reference, "query":query, "response":response})

            except Exception as e:
                results.append({"reference": "", "query":query, "response":str(e)})
        
        df = pd.DataFrame(results)
        df["context"] = df["reference"]
        df.rename(columns={"query": "input", "response": "output"}, inplace=True)
        assert all(column in df.columns for column in ["output", "input", "context", "reference"])
        df.to_csv("artifacts/rag_df.csv", index=False)
        # Run Phoenix evaluations
        print("\n📊 Running Phoenix evaluations...")
        
        queries_df = get_qa_with_reference(px.Client())
        queries_df.to_csv("artifacts/queries_df.csv", index=False)

        retrieved_documents_df = get_retrieved_documents(px.Client())
        retrieved_documents_df.to_csv("artifacts/retrieved_documents_df.csv", index=False)
        # df = pd.read_csv("artifacts/rag_df.csv")
        # 1. Hallucination Detection
        hallucination_evaluator = HallucinationEvaluator(self.eval_model)
        qa_correctness_evaluator = QAEvaluator(self.eval_model)
        

        hallucination_eval_df, qa_eval_df = run_evals(
        dataframe=queries_df, evaluators=[hallucination_evaluator, qa_correctness_evaluator], provide_explanation=True
    )   
        
        hallucination_eval_df.to_csv("artifacts/hallucination_eval_df.csv", index=False)
        qa_eval_df.to_csv("artifacts/qa_eval_df.csv", index=False)

        
        relevance_evaluator = RelevanceEvaluator(self.eval_model)
        relevance_eval_df = run_evals(
            dataframe=queries_df,
            evaluators=[relevance_evaluator],
            provide_explanation=True,
        )[0]
      
        relevance_eval_df.to_csv("artifacts/relevance_eval_df.csv", index=False)

        

    #     
    
    


        
    

# Example usage and test queries
TEST_QUERIES = [
    {
        "query": "What are the procurement standards?",
        "expected_answer": "Abu Dhabi procurement standards include transparency, competitiveness, and value for money."
    }
    # ,
    # {
    #     "query": "How should vendors be evaluated?",
    #     "expected_answer": "Vendors are evaluated based on eligibility, past performance, financial standing, compliance with specifications, and value for money."
    # },
    # {
    #     "query": "What is the approval process for purchases?",
    #     "expected_answer": "Purchase requests must go through defined approval levels based on financial thresholds, with higher-value procurements requiring senior management or committee approval."
    # },
    # {
    #     "query": "What are the main procurement methods?",
    #     "expected_answer": "The main procurement methods are open tendering, restricted tendering, request for quotations, direct procurement, and competitive negotiation."
    # },
    # {
    #     "query": "How should conflicts of interest be handled?",
    #     "expected_answer": "Conflicts of interest must be declared immediately, and staff involved must recuse themselves from related procurement decisions."
    # }
]


async def run_phoenix_evaluation():
    """Run Phoenix evaluation on test queries"""
    evaluator = PhoenixRAGEvaluator()
    
    # Run evaluation
    results_df = await evaluator.evaluate_queries(TEST_QUERIES)
    
    # # Print summary
    # evaluator.print_evaluation_summary(results_df)
    
    # # Save results
    # results_df.to_csv("artifacts/phoenix_evaluation_results.csv", index=False)
    # print(f"\n💾 Results saved to: artifacts/phoenix_evaluation_results.csv")
    
    # return results_df

if __name__ == "__main__":
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Run evaluation
    results = asyncio.run(run_phoenix_evaluation())
    
    