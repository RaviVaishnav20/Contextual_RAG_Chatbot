from pathlib import Path
from typing import List
from typing_extensions import Annotated
from zenml import step

from contextual_rag.application.extractors.resources import convert_to_markdown
from contextual_rag.settings import settings


@step(enable_cache=False)
def save_markdown(files: List[str]) -> List[Path]:
    out_dir = settings.markdown_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in files:
        p = Path(f)
        # If already markdown, copy as-is
        if p.suffix.lower() == ".md":
            md_path = out_dir / p.name
            md_path.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            written.append(md_path)
            continue

        md_text = convert_to_markdown(p)
        md_path = out_dir / f"{p.stem}.md"
        md_path.write_text(md_text, encoding="utf-8")
        written.append(md_path)
    return written
