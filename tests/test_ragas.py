from contextual_rag.model.evaluation.ragas import RagasEvaluator
import asyncio
if __name__=="__main__":
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
    evaluator = RagasEvaluator()
  
    # Run evaluation
    response = asyncio.run(evaluator.evaluate_rag_batch(TEST_QUERIES))
    print("\n🎉 RAGAS evaluation complete!")
    print(f"Check {response} for detailed results")
