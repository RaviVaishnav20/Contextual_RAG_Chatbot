
import argparse

from zenml_config import register_materializers
from pipelines.ingest_resources import ingest_resources
from pipelines.chunking_pipeline import chunking_pipeline
from pipelines.build_embedding import build_embedding
from pipelines.query_pipeline import query_pipeline


def main():
    # Register custom materializers before running any pipelines
    register_materializers()
    
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")
    sub.add_parser("chunking")
    sub.add_parser("embedding")
    q = sub.add_parser("query")
    q.add_argument("question")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_resources()
    elif args.cmd == "chunking":
        chunking_pipeline()
    elif args.cmd == "embedding":
        build_embedding()
    elif args.cmd == "query":
        res = query_pipeline(args.question)
        print(res)


if __name__ == "__main__":
    main()
