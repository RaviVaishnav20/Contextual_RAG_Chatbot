import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os
import asyncio
import pandas as pd
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Optional
from agentic_rag.rag_rerank import get_rag_response
# from config.config_manager import ConfigManager
from ragas import EvaluationDataset
from dotenv import load_dotenv
load_dotenv()
class RAGASEvaluator:
    """Simple RAGAS evaluation and tracing for your RAG system"""
    
    def __init__(self):
        # self.config = ConfigManager()
        
        # Initialize OpenAI models for RAGAS (using GPT-4o-mini for cost efficiency)
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure RAGAS metrics
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        answer_relevancy.embeddings = self.embeddings
        context_precision.llm = self.llm
        context_recall.llm = self.llm

    async def prepare_evaluation_dataset(self, test_queries: List[Dict[str, str]], rag_results_path: Optional[str] = None) -> EvaluationDataset:
            """
            Prepare dataset for RAGAS evaluation
            
            Args:
                test_queries: List of dicts with 'question', 'ground_truth' (optional)
                rag_results_path: Optional path to a CSV file containing pre-run RAG results
            """
            print("🔍 Preparing RAGAS evaluation dataset...")
            
            test_queries_map = {q['question']: q.get('ground_truth', '') for q in test_queries}
            dataset_list = []
            if rag_results_path:
                print(f"Reading RAG results from {rag_results_path}...")
                try:
                    rag_df = pd.read_csv(rag_results_path)
                    for i, row in rag_df.iterrows():
                        question = row['input']
                        dataset_list.append({
                            'user_input': question,
                            'response': row['output'],
                            'retrieved_contexts': self._extract_contexts(row['context']),
                            'reference':test_queries_map.get(question, '')
                        })
                    print(f"Successfully loaded {len(rag_df)} RAG results from CSV.")
                except FileNotFoundError:
                    print(f"Error: RAG results file not found at {rag_results_path}. Proceeding with live RAG calls.")
                    # Fallback to live RAG calls if file not found
                    for i, query_data in enumerate(test_queries):
                        question = query_data['question']
                        ground_truth = query_data.get('ground_truth', '')
                        
                        print(f"Processing query {i+1}/{len(test_queries)}: {question[:50]}...")
                        
                        # Get RAG response
                        rag_result = await get_rag_response(question)
                        
                        # Extract contexts from retrieved text
                        retrieved_contexts = self._extract_contexts(rag_result['retrieved_text'])
 
                        dataset_list.append({
                            'user_input': question,
                            'response': rag_result['llm_response'],
                            'retrieved_contexts': retrieved_contexts,
                            'reference':ground_truth
                        })
            else:
                for i, query_data in enumerate(test_queries):
                    question = query_data['question']
                    ground_truth = query_data.get('ground_truth', '')
                    
                    print(f"Processing query {i+1}/{len(test_queries)}: {question[:50]}...")
                    
                    # Get RAG response
                    rag_result = await get_rag_response(question)
                    
                    # Extract contexts from retrieved text
                    retrieved_contexts = self._extract_contexts(rag_result['retrieved_text'])
                  
                    dataset_list.append({
                            'user_input': question,
                            'response': rag_result['llm_response'],
                            'retrieved_contexts': retrieved_contexts,
                            'reference':ground_truth
                        })
          
            evaluation_dataset = EvaluationDataset.from_list(dataset_list)
            return evaluation_dataset
    
    def _extract_contexts(self, retrieved_text: str) -> List[str]:
        """Extract individual context chunks from retrieved text"""
        if not retrieved_text:
            return ["No context retrieved"]
        
        # Split by chunk separators
        chunks = retrieved_text.split("--- Chunk")
        contexts = []
        
        for chunk in chunks:
            if chunk.strip():
                # Remove metadata and keep only the content
                lines = chunk.split('\n')
                content_lines = []
                skip_next = True  # Skip first line (chunk header)
                
                for line in lines:
                    if skip_next and ('Score:' in line or 'Source:' in line):
                        continue
                    skip_next = False
                    if line.strip():
                        content_lines.append(line.strip())
                
                if content_lines:
                    contexts.append('\n'.join(content_lines))
        
        return contexts if contexts else ["No valid context extracted"]
    
    async def evaluate_rag(self, test_queries: List[Dict[str, str]],rag_results_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run RAGAS evaluation on your RAG system
        
        Args:
            test_queries: List of dicts with 'question' and optional 'ground_truth'
        """
        print("🚀 Starting RAGAS evaluation...")
        
        # Prepare dataset
        dataset = await self.prepare_evaluation_dataset(test_queries,rag_results_path)
        
        # # Define metrics to evaluate
        # metrics = [
        #     faithfulness,
        #     answer_relevancy,
        #     context_precision
        # ]
        
        # # Add context_recall only if ground truths are available
        # if any(dataset['reference'] for dataset in [dataset]):
        #     metrics.append(context_recall)
        
        # print("\n📊 Running RAGAS metrics evaluation...")
        # print(f"Metrics: {[metric.name for metric in metrics]}")
        
        # # Run evaluation
        # result = evaluate(
        #     dataset=dataset,
        #     metrics=metrics,
        # )
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper


        evaluator_llm = LangchainLLMWrapper(self.llm)
        from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

        result = evaluate(dataset=dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
        
        result_df = result.to_pandas()
        result_df.to_csv("artifacts/ragas_evaluate.csv", index=False)
        return result_df
    
    def print_evaluation_summary(self, results_df: pd.DataFrame):
        """Print a summary of RAGAS evaluation results"""
        print("\n" + "="*50)
        print("🎯 RAGAS EVALUATION SUMMARY")
        print("="*50)
        
        metrics = [col for col in results_df.columns if col not in ['user_input','retrieved_contexts','response','reference']]
        
        for metric in metrics:
            if metric in results_df.columns:
                avg_score = results_df[metric].mean()
                print(f"📈 {metric.title()}: {avg_score:.3f}")
        
        print(f"\n📊 Evaluated {len(results_df)} queries")
        
        # Show best and worst performing queries
        if 'answer_relevancy' in results_df.columns:
            best_idx = results_df['answer_relevancy'].idxmax()
            worst_idx = results_df['answer_relevancy'].idxmin()
            
            print(f"\n✅ Best performing query: {results_df.loc[best_idx, 'user_input'][:50]}...")
            print(f"❌ Worst performing query: {results_df.loc[worst_idx, 'user_input'][:50]}...")
    
    def trace_rag_call(self, user_input: str, response: str, retrieved_contexts: List[str]):
        """Simple tracing for debugging RAG calls"""
        print("\n" + "="*40)
        print("🔍 RAG CALL TRACE")
        print("="*40)
        print(f"📝 Question: {user_input}")
        print(f"\n🔎 Retrieved {len(retrieved_contexts)} contexts:")
        for i, ctx in enumerate(retrieved_contexts[:3]):  # Show first 3 contexts
            print(f"  Context {i+1}: {ctx[:100]}...")
        print(f"\n💬 Generated Answer: {response[:200]}...")


# Example usage and test queries
TEST_QUERIES = [
    {
        "question": "What are the procurement standards?",
        "ground_truth": "Abu Dhabi procurement standards include transparency, competitiveness, and value for money."
    }
    ,
    {
        "question": "How should vendors be evaluated?",
        "ground_truth": "Vendors are evaluated based on eligibility, past performance, financial standing, compliance with specifications, and value for money."
    },
    {
        "question": "What is the approval process for purchases?",
        "ground_truth": "Purchase requests must go through defined approval levels based on financial thresholds, with higher-value procurements requiring senior management or committee approval."
    },
    {
        "question": "What are the main procurement methods?",
        "ground_truth": "The main procurement methods are open tendering, restricted tendering, request for quotations, direct procurement, and competitive negotiation."
    },
    {
        "question": "How should conflicts of interest be handled?",
        "ground_truth": "Conflicts of interest must be declared immediately, and staff involved must recuse themselves from related procurement decisions."
    }
]

async def run_ragas_evaluation(TEST_QUERIES, rag_results_path=None):
    """Run RAGAS evaluation on test queries"""
    evaluator = RAGASEvaluator()
    
    # Run evaluation
    results_df = await evaluator.evaluate_rag(TEST_QUERIES, rag_results_path)
    
    # Print summary
    evaluator.print_evaluation_summary(results_df)
    
    # Save results
    results_df.to_csv("artifacts/ragas_evaluation_results.csv", index=False)
    print(f"\n💾 Results saved to: artifacts/ragas_evaluation_results.csv")
    
    # # Example of tracing a single call
    # print("\n" + "="*50)
    # print("🔍 TRACING EXAMPLE")
    # print("="*50)
    
    # sample_question = TEST_QUERIES[0]['question']
    # rag_result = await get_rag_response(sample_question)
    # contexts = evaluator._extract_contexts(rag_result['retrieved_text'])
    
    # evaluator.trace_rag_call(
    #     user_input=sample_question,
    #     response=rag_result['llm_response'],
    #     retrieved_contexts=contexts
    # )
    
    return results_df

if __name__ == "__main__":
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Run evaluation "artifacts/rag_df.csv"
    results = asyncio.run(run_ragas_evaluation(TEST_QUERIES, rag_results_path=None))
    
    print("\n🎉 RAGAS evaluation complete!")
    print("Check artifacts/ragas_evaluation_results.csv for detailed results")
    