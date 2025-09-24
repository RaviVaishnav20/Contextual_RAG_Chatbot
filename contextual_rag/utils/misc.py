import re 


def remove_think_portion(text: str) -> str:
    """Remove <think>...</think> portion from the given text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()