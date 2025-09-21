from contextual_rag.infrastructure.llm import generate_content
from typing import List, Tuple
from contextual_rag.infrastructure.config_manager import ConfigManager

def synthesize_answer(contexts: List[Tuple[Tuple[str, float, str], float]], question: str) -> str:
    # Placeholder answer synthesis
    joined = "\n\n".join(c[0][2] for c in contexts)
    return f"#Generate answer for given Query, refer Context to provide the answer \n\n ##Query: {question} \n\n ##Context: {joined}"



def generate_answer(contexts: List[Tuple[Tuple[str, float, str], float]], question: str) -> str:
    cm = ConfigManager()
    rag_cfg = cm.get_rag_config() or {}
    
    primary_provider = rag_cfg.get('rag_answer', {}).get('primary_provider', 'ollama')
    primary_model = rag_cfg.get('rag_answer', {}).get('primary_model_name', 'llama3:8b')
    fallback_provider = rag_cfg.get('rag_answer', {}).get('fallback_provider', 'gemini')
    fallback_model = rag_cfg.get('rag_answer', {}).get('fallback_model_name', 'gemini-2.5-flash')

    prompt = synthesize_answer(contexts, question)
    # print("prompt")
    # print(prompt)
    
    try:
        answer = generate_content(
            provider=primary_provider,
            model_name=primary_model,
            prompt=prompt
        ).strip()
        print("answer")
        print(type(answer))
        print(answer)
        return answer
    except Exception as e:
        try:
            answer = generate_content(
                provider=fallback_provider,
                model_name=fallback_model,
                prompt=prompt
            ).strip()
            
        except Exception as e:
            return f"Unable generate answer for User query {e}"