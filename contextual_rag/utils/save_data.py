import csv
import os
from contextual_rag.settings import settings
import pandas as pd

def save_rag_response(query: str, answer: str, context: str):
    """Save a single RAG response to CSV (append if exists)."""
    rag_metadata_file = settings.rag_metadata_file
    file_exists = os.path.isfile(rag_metadata_file)
    
    with open(rag_metadata_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output", "context"])
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({"input": query, "output": answer, "context": context})



def save_ragas_response(df: pd.DataFrame) -> str:
    """
    Save RAGAS evaluation responses to CSV.
    Appends if file exists, creates fresh if not.
    
    Expected columns:
    user_input, retrieved_contexts, response, reference,
    context_recall, faithfulness, factual_correctness
    """
    ragas_metadata_file = settings.ragas_metadata_file
    expected_columns = [
        "user_input", "retrieved_contexts", "response", "reference",
        "context_recall", "faithfulness", "factual_correctness(mode=f1)"
    ]
   
    # Validate columns
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    
    file_exists = os.path.isfile(ragas_metadata_file)

    # Append mode if exists, write mode if new
    df.to_csv(
        ragas_metadata_file,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        encoding="utf-8"
    )
    # df.to_csv(ragas_metadata_file, index=False)
    return f"\n💾 Results saved to: {ragas_metadata_file}"