from contextual_rag.model.evaluation.ragas import RagasEvaluator
import asyncio
import json
from contextual_rag.settings import settings
def load_questions(file_path: str):
    """Load Q&A pairs from a JSON file into a Python variable."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__=="__main__":
    file_path = settings.ragas_gt_dataset
    TEST_QUERIES = load_questions(file_path)


    evaluator = RagasEvaluator()
  
    # Run evaluation
    response = asyncio.run(evaluator.evaluate_rag_batch(TEST_QUERIES))
    print("\n🎉 RAGAS evaluation complete!")
    print(f"Check {response} for detailed results")
