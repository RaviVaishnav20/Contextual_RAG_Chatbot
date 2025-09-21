
"""
Test harness to validate individual document-conversion methods independently.

Methods covered:
- md_passthrough: Copy/read .md file directly
- lc_docling: langchain_docling DoclingLoader (plugins enabled, OCR disabled if possible)
- docling_direct: direct Docling DocumentConverter (OCR disabled when possible)
- pymupdf_pdf: PDF-only fallback using pymupdf4llm

Usage examples:
  uv run python tests/test_conversion_methods.py --list
  uv run python tests/test_conversion_methods.py --file resources/HR\ Bylaws.pdf --method lc_docling
  uv run python tests/test_conversion_methods.py --all  # run all methods on all resources
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = PROJECT_ROOT / "resources"
OUTPUT_DIR = PROJECT_ROOT / "data" / "markdown"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TestResult:
    method: str
    file: Path
    success: bool
    content_len: int = 0
    error: Optional[str] = None
    out_path: Optional[Path] = None


def md_passthrough(file_path: Path) -> str:
    if file_path.suffix.lower() != ".md":
        raise ValueError("md_passthrough only supports .md files")
    return file_path.read_text(encoding="utf-8")


def lc_docling_convert(file_path: Path) -> str:
    # Encourage Docling to allow plugins and disable OCR via environment.
    os.environ.setdefault("DOCLING_ALLOW_EXTERNAL_PLUGINS", "true")
    os.environ.setdefault("DOCLING_DO_OCR", "false")
    os.environ.setdefault("DOCLING_OCR_ENABLED", "false")
    os.environ.setdefault("DOCLING_OCR_ENGINE", "none")

    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    try:
        loader = DoclingLoader(
            file_path=str(file_path),
            export_type=ExportType.MARKDOWN,
            allow_external_plugins=True,  # may be ignored depending on version
            ocr_engine=None,              # may be ignored depending on version
        )
    except TypeError:
        loader = DoclingLoader(file_path=str(file_path), export_type=ExportType.MARKDOWN)

    docs = loader.load()
    return "".join(doc.page_content for doc in docs)


def docling_direct_convert(file_path: Path) -> str:
    # Best-effort: try to disable OCR and enable plugins. If options are not available, fall back.
    try:
        from docling.document_converter import (
            DocumentConverter,
            DocumentConverterOptions,  # may not exist on older versions
        )
        from docling_core.types.doc import OcrOptions  # may not exist on older versions

        options = DocumentConverterOptions(
            allow_external_plugins=True,
            ocr_options=OcrOptions(enabled=False),
        )
        converter = DocumentConverter(options=options)
    except Exception:
        from docling.document_converter import DocumentConverter  # type: ignore
        converter = DocumentConverter()

    result = converter.convert(str(file_path))
    return result.document.export_to_markdown()


def pymupdf_pdf_convert(file_path: Path) -> str:
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("pymupdf_pdf only supports .pdf files")
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(file_path))


METHODS: dict[str, Callable[[Path], str]] = {
    "md_passthrough": md_passthrough,
    "lc_docling": lc_docling_convert,
    "docling_direct": docling_direct_convert,
    "pymupdf_pdf": pymupdf_pdf_convert,
}


def run_on_file(method_name: str, file_path: Path) -> TestResult:
    func = METHODS[method_name]
    try:
        logger.info(f"[{method_name}] Converting: {file_path}")
        content = func(file_path)
        out_name = f"{file_path.stem}.{method_name}.md"
        out_path = OUTPUT_DIR / out_name
        out_path.write_text(content, encoding="utf-8")
        logger.info(f"[{method_name}] Wrote: {out_path} ({len(content)} chars)")
        return TestResult(method=method_name, file=file_path, success=True, content_len=len(content), out_path=out_path)
    except Exception as e:
        logger.error(f"[{method_name}] Failed for {file_path}: {e}")
        return TestResult(method=method_name, file=file_path, success=False, error=str(e))


def discover_resources() -> list[Path]:
    files: list[Path] = []
    if RESOURCES_DIR.exists():
        for p in RESOURCES_DIR.rglob("*"):
            if p.is_file():
                files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to a single file to test")
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS.keys()), help="Single method to run")
    parser.add_argument("--all", action="store_true", help="Run all methods on all resources")
    parser.add_argument("--list", action="store_true", help="List discovered resource files")
    args = parser.parse_args()

    if args.list:
        for p in discover_resources():
            print(p)
        return

    results: list[TestResult] = []

    if args.file and args.method:
        results.append(run_on_file(args.method, Path(args.file)))
    elif args.file and not args.method:
        fp = Path(args.file)
        for m in METHODS:
            # only run compatible methods
            if m == "md_passthrough" and fp.suffix.lower() != ".md":
                continue
            if m == "pymupdf_pdf" and fp.suffix.lower() != ".pdf":
                continue
            results.append(run_on_file(m, fp))
    elif args.all:
        for fp in discover_resources():
            for m in METHODS:
                if m == "md_passthrough" and fp.suffix.lower() != ".md":
                    continue
                if m == "pymupdf_pdf" and fp.suffix.lower() != ".pdf":
                    continue
                results.append(run_on_file(m, fp))
    else:
        parser.error("Provide --file (with optional --method) or --all, or use --list")

    # Summary
    print("\n=== Summary ===")
    for r in results:
        status = "OK" if r.success else "FAIL"
        extra = f" len={r.content_len}" if r.success else f" error={r.error}"
        print(f"[{status}] {r.method} :: {r.file.name}{extra}")


if __name__ == "__main__":
    main()


