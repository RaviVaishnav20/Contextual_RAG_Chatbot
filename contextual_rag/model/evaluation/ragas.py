from contextual_rag.settings import settings
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
# from agentic_rag.rag_rerank import get_rag_response
# from config.config_manager import ConfigManager
from contextual_rag.application.rag.rag import get_rag_answer
from ragas import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from contextual_rag.utils.save_data import save_ragas_response
from dotenv import load_dotenv
load_dotenv()
class RagasEvaluator:
    """Simple RAGAS evaluation and tracing for your RAG system"""
    
    def __init__(self):
        # self.config = ConfigManager()
        
        # Initialize OpenAI models for RAGAS (using GPT-4o-mini for cost efficiency)
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano", #gpt-5-nano
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        # Configure RAGAS metrics
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        answer_relevancy.embeddings = self.embeddings
        context_precision.llm = self.llm
        context_recall.llm = self.llm

    async def prepare_evaluation_dataset_with_rag(self, test_queries: List[Dict[str, str]]) -> EvaluationDataset:
        """
        Prepare dataset for RAGAS evaluation
        """
        print("🔍 Preparing RAGAS evaluation dataset...")
        dataset_list = []
            
        for i, query_data in enumerate(test_queries):
            question = query_data['question']
            ground_truth = query_data.get('ground_truth', '')         
           
 
            results = await get_rag_answer(question)
            
            dataset_list.append(SingleTurnSample(
                user_input = question,
                retrieved_contexts = results.retrieved_contexts,
                response = results.answer,
                reference = ground_truth
            ))
        # print("dataset_list")
        # print(dataset_list)
        evaluation_dataset = EvaluationDataset(dataset_list)
        return evaluation_dataset

    def prepare_evaluation_dataset(self, query: str, response: str, contexts: List[str], reference: str = "") -> EvaluationDataset:
        """
        Prepare dataset for RAGAS evaluation
        """
        print("🔍 Preparing RAGAS evaluation dataset...")
       
        evaluation_dataset = EvaluationDataset([SingleTurnSample(
                user_input = query,
                retrieved_contexts = contexts,
                response = response,
                reference = reference
            )])
        return evaluation_dataset

    def evaluate_rag(self, query: str, response: str, contexts: List[str], reference: str = "") -> str:
        """
        Run RAGAS evaluation on your RAG system
        """
        print("🚀 Starting RAGAS evaluation...")
        
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(query, response, contexts, reference)
        result = evaluate(dataset=dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        # result = evaluate(dataset=dataset,metrics=[Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        
        result_df = result.to_pandas()
        # result_df.to_csv("test.csv", index=False)
        response = save_ragas_response(result_df)
        return response

    async def evaluate_rag_batch(self, test_queries: List[Dict[str, str]]) -> str:
        """
        Run RAGAS evaluation on your RAG system
        """
        print("🚀 Starting RAGAS evaluation...")
        
        # Prepare dataset
        dataset = await self.prepare_evaluation_dataset_with_rag(test_queries)
        
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
        

        result = evaluate(dataset=dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        # result = evaluate(dataset=dataset,metrics=[Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        
        result_df = result.to_pandas()
        ragas_evaluation_report = settings.ragas_evaluation_report
        # result_df.to_csv(ragas_evaluation_report, index=False)
        result_df.to_excel(ragas_evaluation_report, sheet_name="Sheet1")
        return f"\n💾 Results saved to: {ragas_evaluation_report}"

    
