#!/usr/bin/env python3
"""
Test script for document conversion using langchain_docling.
Based on the reference implementation provided by the user.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
import os
from loguru import logger

def convert_to_markdown(file_path: str, markdown_artifacts_folder: str) -> str:
    """Convert document to markdown and save to file"""
    try:
        logger.info(f"Converting document: {file_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(markdown_artifacts_folder, exist_ok=True)
        
        # Use langchain_docling to convert
        loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
        docs = loader.load()
        
        # Generate output file path
        file_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(file_name)
        output_file_path = os.path.join(markdown_artifacts_folder, f"{base_name}.md")
        
        # Write markdown content to file
        full_markdown_content = ""
        with open(output_file_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.page_content)
                full_markdown_content += doc.page_content
        
        logger.info(f"Successfully converted and saved to: {output_file_path}")
        return full_markdown_content
        
    except Exception as e:
        logger.error(f"Failed to convert {file_path}: {e}")
        raise

def main():
    """Main function to test document conversion"""
    # Set up paths
    project_root = Path(__file__).resolve().parent
    resources_dir = project_root / "resources"
    markdown_artifacts_folder = project_root / "data" / "markdown"
    
    # Find PDF files in resources directory
    pdf_files = list(resources_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {resources_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to convert")
    
    # Convert each PDF file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            markdown_content = convert_to_markdown(str(pdf_file), str(markdown_artifacts_folder))
            logger.info(f"Converted {pdf_file.name} - Content length: {len(markdown_content)} characters")
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
