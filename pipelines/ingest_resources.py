from zenml import pipeline
from steps.extract.load_resources import load_resources
from steps.extract.save_markdown import save_markdown


@pipeline
def ingest_resources():
    files = load_resources()
    md_paths = save_markdown(files)
    return md_paths
