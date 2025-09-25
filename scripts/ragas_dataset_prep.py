import pandas as pd
import json
from contextual_rag.settings import settings
# Load CSV
csv_file = settings.rag_metadata_file
df = pd.read_csv(csv_file)

# Load JSON
json_file = settings.ragas_gt_dataset
with open(json_file, "r") as f:
    data = json.load(f)

# Convert JSON list to dict for fast lookup
qa_dict = {item["question"]: item["ground_truth"] for item in data}

# Map ground truth from JSON to CSV input column
df["ground_truth"] = df["input"].map(qa_dict)

# Save back to CSV
df.to_csv(settings.ragas_gt_dataset_with_response, index=False)
print("✅ Updated CSV saved as updated_data.csv")
