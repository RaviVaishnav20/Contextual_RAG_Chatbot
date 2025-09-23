from pathlib import Path
from typing import List
from typing_extensions import Annotated
from zenml import step

from contextual_rag.settings import settings
#note: todo file hashing

@step(enable_cache=False)
def load_markdowns() -> List[str]:
    resources_dir = settings.markdown_dir
    files: list[str] = []
    if resources_dir.exists():
        for p in resources_dir.rglob("*"):
            if p.is_file():
                files.append(str(p.resolve()))
    return files
