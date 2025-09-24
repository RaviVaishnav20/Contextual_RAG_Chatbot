from contextual_rag.infrastructure.llm import generate_content
from typing import List, Tuple
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.utils.misc import remove_think_portion
from contextual_rag.application.rag.rag_model import RerankedOutput, RagAnswer

#### if have chunk_context
# def synthesize_prompt(contexts: List[RerankedOutput], question: str) -> Tuple[str, List[str], List[str]]:
#     # Placeholder answer synthesis
#     clean_context = [] 
#     sources = []
#     for c in contexts:
#         clean_context.append(f"Chunk Content: {c.retriever_output.chunk_content} \n Chunk Context: {c.retriever_output.chunk_context}") #(str(c[0][2]).replace('chunk',''))
#         sources.append(str(c.retriever_output.source).replace('.md',''))
#     joined = "\n\n".join(clean_context)
#     prompt = f"""
# # Role: You are an advanced assistant that answers user queries using retrieved document chunks.

# # Instructions:
# - Each retrieved chunk comes with two parts:
#     1. Chunk Content – the actual piece of text.
#     2. Chunk Context – metadata or summary that explains the role, importance, or relationship of the chunk within the larger document.
# - When answering the query:
#     Always ground your answer in the chunk content.
#     Use the chunk context to correctly interpret the meaning, placement, and intent of the chunk within the document.
#     If multiple chunks are provided, combine their information coherently while respecting their context.
#     Be precise, factual, and avoid adding knowledge not supported by the chunks or their context.
#     If the answer cannot be determined from the given chunks and context, clearly state that.

# # Query:
# {question}
# \n\n
# # Context:
# {joined}
# """
#     return (prompt , clean_context, list(set(sources)))

def synthesize_prompt(contexts: List[RerankedOutput], question: str) -> Tuple[str, List[str], List[str]]:
    # Placeholder answer synthesis
    clean_context = [] 
    sources = []
    for c in contexts:
        clean_context.append(f"{c.retriever_output.chunk_content}") #(str(c[0][2]).replace('chunk',''))
        sources.append(str(c.retriever_output.source).replace('.md',''))
    joined = "\n\n".join(clean_context)
    prompt = f"""
# Role: You are an advanced assistant that answers user queries using retrieved document chunks.

# Instructions:
- When answering the query correctly interpret the meaning, placement, and intent:
- Use the context to correctly answer query.
- Be precise, factual, and avoid adding knowledge not supported by the context.
- If the answer cannot be determined from the given context, clearly state that.

# Query:
{question}
\n\n
# Context:
{joined}
"""
    return (prompt , clean_context, list(set(sources)))

def generate_answer(contexts: List[RerankedOutput], question: str) -> RagAnswer:
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    
    primary_provider = rag_cfg.get('rag_answer', {}).get('primary_provider', 'ollama')
    primary_model = rag_cfg.get('rag_answer', {}).get('primary_model_name', 'llama3:8b')
    fallback_provider = rag_cfg.get('rag_answer', {}).get('fallback_provider', 'gemini')
    fallback_model = rag_cfg.get('rag_answer', {}).get('fallback_model_name', 'gemini-2.5-flash')

    from contextual_rag.application.rag.generate_answer import synthesize_prompt
    synt_prompt = synthesize_prompt(contexts, question)
    prompt, clean_context, sources = synt_prompt

    try:
        answer = generate_content(
            provider=primary_provider,
            model_name=primary_model,
            prompt=prompt
        ).strip()
        answer = remove_think_portion(answer)

        return RagAnswer(
            answer=answer,
            retrieved_contexts=clean_context,
            sources=sources
        )

    except Exception:
        try:
            answer = generate_content(
                provider=fallback_provider,
                model_name=fallback_model,
                prompt=prompt
            ).strip()
            answer = remove_think_portion(answer)

            return RagAnswer(
                answer=answer,
                retrieved_contexts=clean_context,
                sources=sources
            )

        except Exception as e:
            return RagAnswer(
                answer=f"Unable to generate answer for User query: {e}",
                retrieved_contexts=clean_context,
                sources=sources
            )