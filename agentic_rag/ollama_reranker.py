import requests
from typing import List, Tuple
import re

class OllamaReRanker:
    """Simple Ollama-based re-ranker for RAG pipeline"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return "0.5"
    
    def _score_document(self, query: str, document: str) -> float:
        """Score a single document for relevance"""
        prompt = f"""Rate how relevant this document is to the query on a scale of 0.0 to 1.0.
Return ONLY the numeric score (e.g., 0.85).

Query: {query}

Document: {document[:800]}

Score:"""
        
        try:
            score_text = self._call_ollama(prompt)
            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                # Normalize if needed
                if score > 1.0:
                    score = score / 10.0 if score <= 10 else score / 100.0
                return max(0.0, min(1.0, score))
        except:
            pass
        return 0.5  # Default score
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Re-rank documents by relevance"""
        if not documents:
            return []
        
        print(f"Re-ranking {len(documents)} documents with {self.model_name}...")
        
        # Score each document
        doc_scores = []
        for doc in documents:
            score = self._score_document(query, doc)
            doc_scores.append((doc, score))
        
        # Sort by score and return top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k] if top_k else doc_scores
    


if __name__ == "__main__":
    reranker = OllamaReRanker()

    query = "What are the new HR policies regarding remote work?"
    documents = [
        "This document discusses the company's Q3 financial results.",
        "New HR policies have been released, detailing flexible work arrangements and remote work guidelines.",
        "A memo about the upcoming company picnic and team-building activities.",
        "This report summarizes the annual performance reviews for all employees.",
        "Guidelines for submitting expense reports and travel reimbursements."
    ]

    print(f"\nOriginal Documents:\n{'- ' * 15}")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc[:100]}...") # Print first 100 chars for brevity

    reranked_docs = reranker.rerank(query, documents, top_k=3)

    print(f"\nRe-ranked Documents (Top {len(reranked_docs)}):\n{'- ' * 15}")
    for doc, score in reranked_docs:
        print(f"Score: {score:.4f} - {doc[:100]}...")