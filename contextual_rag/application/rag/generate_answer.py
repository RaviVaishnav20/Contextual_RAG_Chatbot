from contextual_rag.infrastructure.llm import generate_content
from typing import List, Tuple
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.utils.misc import remove_think_portion
def synthesize_answer(contexts: List[Tuple[Tuple[str, float, str, str], float]], question: str) -> Tuple[str, List[str], List[str]]:
    # Placeholder answer synthesis
    clean_context = [] # [c[0][2] for c in contexts]
    sources = []
    for c in contexts:
        clean_context.append(str(c[0][2]).replace('chunk',''))
        sources.append(str(c[0][3]).replace('.md',''))
    joined = "\n\n".join(clean_context)


    return (f"""
You are an expert assistant. Answer the user's Query directly and clearly, 
using the provided Context only as supporting information. 
'

## Query:
{question}

## Context:
{joined}
""", clean_context, list(set(sources)))



def generate_answer(contexts: List[Tuple[Tuple[str, float, str, str], float]], question: str) -> Tuple[str, List[str], List[str]]:
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    
    primary_provider = rag_cfg.get('rag_answer', {}).get('primary_provider', 'ollama')
    primary_model = rag_cfg.get('rag_answer', {}).get('primary_model_name', 'llama3:8b')
    fallback_provider = rag_cfg.get('rag_answer', {}).get('fallback_provider', 'gemini')
    fallback_model = rag_cfg.get('rag_answer', {}).get('fallback_model_name', 'gemini-2.5-flash')

    synt_response = synthesize_answer(contexts, question)
    prompt = synt_response[0]
    clean_context = synt_response[1] 
    sources = synt_response[2] 
    # print("prompt")
    # print(prompt)
    
    try:
        answer = generate_content(
            provider=primary_provider,
            model_name=primary_model,
            prompt=prompt
        ).strip()
        # print("answer")
        # print(type(answer))
        # print(answer)
        answer = remove_think_portion(answer)
        return (answer, clean_context, sources)
    except Exception as e:
        try:
            answer = generate_content(
                provider=fallback_provider,
                model_name=fallback_model,
                prompt=prompt
            ).strip()
            answer = remove_think_portion(answer)
            return (answer, clean_context, sources)
        except Exception as e:
            return (f"Unable generate answer for User query {e}", [""])