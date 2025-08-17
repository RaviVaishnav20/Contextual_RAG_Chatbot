import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
import os
from config.config_manager import ConfigManager

def convert_to_markdown(file_path: str, markdown_artifacts_folder: str) -> str:
    """Convert document to markdown and save to file"""
    loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
    docs = loader.load()
    
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    output_file_path = os.path.join(markdown_artifacts_folder, f"{base_name}.md")
    
    full_markdown_content = ""
    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content)
            full_markdown_content += doc.page_content
    
    return full_markdown_content



if __name__ == "__main__":
    config = ConfigManager()
    FILE_PATH = config.get('directories.resources') + "/Abu Dhabi Procurement Standards.PDF"
    MARKDOWN_ARTIFACTS_FOLDER = config.get('directories.markdown_artifacts')
    
    os.makedirs(MARKDOWN_ARTIFACTS_FOLDER, exist_ok=True)
    
    # Test markdown conversion
    full_markdown_doc = convert_to_markdown(FILE_PATH, MARKDOWN_ARTIFACTS_FOLDER)
  