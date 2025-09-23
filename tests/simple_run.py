#!/usr/bin/env python3
"""
Simple runner without ZenML for testing the core functionality
"""
import argparse
from pathlib import Path

from contextual_rag.application.extractors.resources import extract_resources_to_markdown
from contextual_rag.application.preprocessing.chunking_data_handlers import build_chunks
from contextual_rag.application.preprocessing.embedding_data_handlers import upsert_pgvector
from contextual_rag.application.rag.retriever import retrieve
from contextual_rag.application.rag.reranking import rerank


def embed_query(text: str) -> list[float]:
    # Placeholder query embedding
    import random
    random.seed(1)
    return [random.random() for _ in range(8)]


def synthesize_answer(contexts: list[tuple[str, float, str]], question: str) -> str:
    # Placeholder answer synthesis
    joined = "\n\n".join(c[2] for c in contexts)
    return f"Q: {question}\nA (stub using {len(contexts)} contexts):\n{joined[:1000]}"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")
    sub.add_parser("index")
    q = sub.add_parser("query")
    q.add_argument("question")

    args = parser.parse_args()

    if args.cmd == "ingest":
        print("Extracting resources to markdown...")
        md_paths = extract_resources_to_markdown()
        print(f"Extracted {len(md_paths)} markdown files")
        for path in md_paths:
            print(f"  - {path}")
            
    elif args.cmd == "index":
        print("Building chunks...")
        chunks = build_chunks()
        print(f"Created {len(chunks)} chunks")
        
        print("Embedding and storing chunks...")
        n_embedded = upsert_pgvector(chunks)
        print(f"Embedded and stored {n_embedded} chunks")
        
    elif args.cmd == "query":
        print(f"Querying: {args.question}")
        
        # Embed query
        qvec = embed_query(args.question)
        
        # Retrieve
        candidates = retrieve(qvec, top_k=5)
        print(f"Retrieved {len(candidates)} candidates")
        
        # Rerank
        ranked = rerank(candidates, top_k=3)
        print(f"Reranked to {len(ranked)} results")
        
        # Answer
        answer = synthesize_answer(ranked, args.question)
        print("\n" + "="*50)
        print(answer)
        print("="*50)


if __name__ == "__main__":
    main()
