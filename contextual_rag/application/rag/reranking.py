
from typing import List, Tuple
import re
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.infrastructure.llm import generate_content
from tqdm import tqdm
class CandidateScorer:
    """Simple Ollama-based re-ranker for RAG pipeline"""
    
    def __init__(self):
        cm = ConfigManager()
        rag_cfg = cm.get_rag_config() or {}
        
        self.primary_provider = rag_cfg.get('reranker', {}).get('primary_provider', 'ollama')
        self.primary_model = rag_cfg.get('reranker', {}).get('primary_model_name', 'llama3:8b')
        self.fallback_provider = rag_cfg.get('reranker', {}).get('fallback_provider', 'gemini')
        self.fallback_model = rag_cfg.get('reranker', {}).get('fallback_model_name', 'gemini-2.5-flash')

    
    def _score_candidate(self, query: str, candidate: str) -> float:
        if len(candidate) > 800:
            candidate = candidate[:800]

        """Score a single document for relevance"""
        prompt = f"""Rate how relevant this document is to the query on a scale of 0.0 to 1.0.
Return ONLY the numeric score (e.g., 0.85).

Query: {query}

Document: {candidate}

Score:"""
        
        try:
            score_text = generate_content(
                provider=self.primary_provider,
                model_name=self.primary_model,
                prompt=prompt
            ).strip()
            print(f"score_text: {score_text}")
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                # Normalize if needed
                if score > 1.0:
                    score = score / 10.0 if score <= 10 else score / 100.0
                return max(0.0, min(1.0, score))
            else:
                return 0.5
        except Exception as e:
            try:
                score_text = generate_content(
                    provider=self.fallback_provider,
                    model_name=self.fallback_model,
                    prompt=prompt
                ).strip()
                
                score_match = re.search(r'(\d+\.?\d*)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    # Normalize if needed
                    if score > 1.0:
                        score = score / 10.0 if score <= 10 else score / 100.0
                    return max(0.0, min(1.0, score))
                else:
                    return 0.5
            except:
                return 0.5

        
def rerank(query: str, candidates: List[Tuple[str, float, str]]) -> List[Tuple[Tuple[str, float, str], float]]:
    """Re-rank documents by relevance"""
    if not candidates:
        return []
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    top_k = rag_cfg.get('reranker', {}).get('top_k','')
    scorer = CandidateScorer()
    print(f"Re-ranking {len(candidates)} candidates...")
    
    # Score each document
    candidate_scores = []
    for candidate in tqdm(candidates):
        score = scorer._score_candidate(query, candidate[2])
        candidate_scores.append((candidate, score))
   
    # Sort by score and return top_k
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    return candidate_scores[:top_k] if top_k else candidate_scores