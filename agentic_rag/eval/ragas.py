import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
from typing import Dict, List
# RAGAS imports
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# RAGAS evaluator setup
class RAGASEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",  # Cost-effective model
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        
        # Configure RAGAS metrics
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        context_precision.llm = self.llm
    
    async def evaluate_response(self, query: str, response: str, contexts: List[str], reference: str = "") -> Dict[str, float]:
        """Evaluate a single RAG response using RAGAS metrics"""
        try:
            dataset = EvaluationDataset.from_list([{
                'user_input': query,
                'response': response,
                'retrieved_contexts': contexts,
                'reference': reference
            }])
            
            metrics = [faithfulness, answer_relevancy]
            if reference:
                metrics.append(context_precision)
            
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.evaluator_llm
            )
            
            return result.to_pandas().iloc[0].to_dict()
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {"error": str(e)}