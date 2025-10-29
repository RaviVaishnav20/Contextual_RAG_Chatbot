
from typing import List, Tuple
import re
from contextual_rag.infrastructure.config_manager import ConfigManager
# from contextual_rag.application.networks.llm import generate_content
from tqdm import tqdm
from contextual_rag.utils.misc import remove_think_portion
from contextual_rag.application.rag.rag_model import RetrieverOutput, RerankedOutput
from contextual_rag.application.networks.embeddings import CrossEncoderModelSingleton
# class CandidateScorer:
#     """Simple Ollama-based re-ranker for RAG pipeline"""
    
#     def __init__(self):
#         cm = ConfigManager()
#         rag_cfg = cm.get_rag_config() or {}
        
#         self.primary_provider = rag_cfg.get('reranker', {}).get('primary_provider', 'ollama')
#         self.primary_model = rag_cfg.get('reranker', {}).get('primary_model_name', 'llama3:8b')
#         self.fallback_provider = rag_cfg.get('reranker', {}).get('fallback_provider', 'gemini')
#         self.fallback_model = rag_cfg.get('reranker', {}).get('fallback_model_name', 'gemini-2.5-flash')

    
#     def _score_candidate(self, query: str, candidate: str) -> float:
#         if len(candidate) > 800:
#             candidate = candidate[:800]

#         """Score a single document for relevance"""
#         prompt = f"""Rate how relevant this document is to the query on a scale of 0.0 to 1.0.
# Return ONLY the numeric score (e.g., 0.85).

# Query: {query}

# Document: {candidate}

# Score:"""
        
#         try:
#             score_text = generate_content(
#                 provider=self.primary_provider,
#                 model_name=self.primary_model,
#                 prompt=prompt
#             ).strip()
#             score_text = remove_think_portion(score_text)
#             # print(f"score_text: {score_text}")
#             score_match = re.search(r'(\d+\.?\d*)', score_text)
#             if score_match:
#                 score = float(score_match.group(1))
#                 # Normalize if needed
#                 if score > 1.0:
#                     score = score / 10.0 if score <= 10 else score / 100.0
#                 return max(0.0, min(1.0, score))
#             else:
#                 return 0.5
#         except Exception as e:
#             try:
#                 score_text = generate_content(
#                     provider=self.fallback_provider,
#                     model_name=self.fallback_model,
#                     prompt=prompt
#                 ).strip()
#                 score_text = remove_think_portion(score_text)
#                 score_match = re.search(r'(\d+\.?\d*)', score_text)
#                 if score_match:
#                     score = float(score_match.group(1))
#                     # Normalize if needed
#                     if score > 1.0:
#                         score = score / 10.0 if score <= 10 else score / 100.0
#                     return max(0.0, min(1.0, score))
#                 else:
#                     return 0.5
#             except:
#                 return 0.5

## uncomment this and comment below rerank function to enable reranking   
# def rerank(query: str, candidates: RetrieverOutput) -> List[RerankedOutput]:
#     """Re-rank documents by relevance"""
#     if not candidates:
#         return []
#     cm = ConfigManager()
#     rag_cfg = cm.get_rag_config() or {}
#     top_k = rag_cfg.get('reranker', {}).get('top_k','')
#     scorer = CandidateScorer()
#     # print(f"Re-ranking {len(candidates)} candidates...")
    
#     # Score each document
#     candidate_scores = []
#     for candidate in candidates:
#         score = scorer._score_candidate(query, candidate.chunk_content)
#         candidate_scores.append((candidate, score))
   
#     # Sort by score and return top_k
#     candidate_scores.sort(key=lambda x: x[1], reverse=True)
#     raw_reranked = candidate_scores[:top_k] if top_k else candidate_scores
#     # Convert to Pydantic
#     reranked_results: List[RerankedOutput] = [
#         RerankedOutput(
#             retriever_output=candidate,
#             rerank_score=rerank_score,
#         )
#         for candidate, rerank_score in raw_reranked
#     ]
#     return reranked_results


# def rerank(query: str, candidates: RetrieverOutput) -> List[RerankedOutput]:
#     """Re-rank documents by relevance"""
    
#     print(f"Re-ranking {len(candidates)} candidates...")
    
#     # Score each document
#     candidate_scores = []
#     for candidate in candidates:
#         candidate_scores.append((candidate, 0.5))
   
#     # Sort by score and return top_k
#     candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
#     # Convert to Pydantic
#     reranked_results: List[RerankedOutput] = [
#         RerankedOutput(
#             retriever_output=candidate,
#             rerank_score=rerank_score,
#         )
#         for candidate, rerank_score in candidate_scores
#     ]
#     return reranked_results


### Reranking cross-encoder

def rerank(query: str, candidates: RetrieverOutput) -> List[RerankedOutput]:
    """Re-rank documents by relevance"""
    
    print(f"Re-ranking {len(candidates)} candidates...")
    if not candidates:
        return []
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    keep_top_k = rag_cfg.get('reranker', {}).get('top_k','')
    _model = CrossEncoderModelSingleton()
    # Score each document
    # candidate_scores = []
    # for candidate in candidates:
    #     candidate_scores.append((candidate, 0.5))

    query_doc_tuples = [(query, chunk.chunk_content) for chunk in candidates]
    scores = _model(query_doc_tuples)
    print(scores)

    scored_query_doc_tuples = list(zip(candidates,scores,strict=False))
    scored_query_doc_tuples.sort(key=lambda x: x[1], reverse=True)

    reranked_documents = scored_query_doc_tuples[:keep_top_k]
    
    reranked_results: List[RerankedOutput] = [
        RerankedOutput(
            retriever_output=candidate,
            rerank_score=rerank_score,
        )
        for candidate, rerank_score in reranked_documents
    ]
    return reranked_results
