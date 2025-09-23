import json
import os
from tqdm import tqdm
def convert_format(input_file, output_file):
    # Load first format JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    converted = []
    for item in tqdm(data):
        # Get file name and replace extension with .md
        source_file = os.path.splitext(item["metadata"]["source"])[0] + ".md"

        # Build new structure
        new_item = {
            "document_name": source_file,
            "chunk_id": f"chunk-{item['metadata']['chunk_id']}",
            "text": item["metadata"]["contextual_chunk_content"],  # take contextual content
            "metadata": {
                "source": "markdown"
            }
        }
        converted.append(new_item)

    # Save as new JSON
    with open(output_file, "w") as f:
        json.dump(converted, f, indent=4)

# Example usage
if __name__ == "__main__":
    convert_format("metadata_context_chunk.json", "output.json")
