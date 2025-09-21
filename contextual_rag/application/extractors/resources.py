from pathlib import Path
from typing import Iterable

from loguru import logger
from contextual_rag.settings import settings


# SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".html", ".md"}


# def iter_supported_files(root: Path) -> Iterable[Path]:
#     for p in root.rglob("*"):
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
#             yield p


def convert_to_markdown(path: Path) -> str:
    """Convert document to markdown using langchain_docling with proper error handling."""
    try:
        from langchain_docling import DoclingLoader
        from langchain_docling.loader import ExportType
       
        logger.info(f"Converting document: {path}")
        
        # Try langchain_docling first with plugins enabled and OCR disabled
        try:
            
            loader = DoclingLoader(file_path=path, export_type=ExportType.MARKDOWN)
            docs = loader.load()
            content = "".join(doc.page_content for doc in docs)
            logger.info(f"Successfully converted {path} using langchain_docling")
            return content
        except Exception as e:
            logger.warning(f"langchain_docling failed for {path}: {e}")
        
    except Exception as e:
        logger.warning(f"Langchain Docling failed for {path} with error: {e}. Trying PDF-specific fallback.")
        try:
            # PDF-specific lightweight fallback
            if path.suffix.lower() == ".pdf":
                import pymupdf4llm
                return pymupdf4llm.to_markdown(str(path))
        except Exception as e2:
            logger.error(f"Fallback PDF conversion also failed for {path}: {e2}")
        # Final safe fallback
        return f"# Error converting document\n\nFailed to convert {path.name} to markdown."


# def extract_resources_to_markdown(resources_dir: Path | None = None) -> list[Path]:
#     resources_dir = resources_dir or (settings.project_root / "resources")
#     out_dir = settings.markdown_dir
#     out_dir.mkdir(parents=True, exist_ok=True)

#     if not resources_dir.exists():
#         logger.warning(f"Resources directory not found: {resources_dir}")
#         return []

#     written: list[Path] = []
#     for file_path in iter_supported_files(resources_dir):
#         md_text = convert_to_markdown(file_path)
#         md_path = out_dir / f"{file_path.stem}.md"
#         md_path.write_text(md_text, encoding="utf-8")
#         written.append(md_path)
#         logger.info(f"Wrote markdown: {md_path}")

#     return written
