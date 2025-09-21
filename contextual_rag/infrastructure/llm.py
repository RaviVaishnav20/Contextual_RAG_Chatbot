import os
import json
import logging
from typing import List

import numpy as np
import requests

from contextual_rag.infrastructure.config_manager import ConfigManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)


def _get_ollama_host() -> str:
    # Prefer explicit docker host if present
    cm = ConfigManager()
    return cm.get_ollama_host()


def generate_content(provider: str, model_name: str, prompt: str) -> str:
    """
    Unified text generation across providers: gemini, groq, ollama, bedrock.
    """
    prov = (provider or "").lower()
    if prov == "bedrock":
        import boto3
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.95,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        model_id = os.getenv("MODEL_ID")
        resp = client.invoke_model(modelId=model_id, body=json.dumps(request))
        body = json.loads(resp.get("body").read())
        return body["content"][0]["text"]

    if prov == "gemini":
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        r = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(r, "text", "") or ""

    if prov == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model_name
        )
        return r.choices[0].message.content

    if prov == "ollama":
        url = f"{_get_ollama_host()}/api/chat"
        # We support both prompt and messages; default to messages for local usage
        resp = requests.post(
            url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Some Ollama frontends return {message: {content}}; others return {response}
        return (
            (data.get("message") or {}).get("content")
            or data.get("response", "")
        )

    raise ValueError("Unsupported provider. Use 'gemini', 'groq', 'ollama', or 'bedrock'.")


def get_embedding(text: str, provider:str, model:str, dim:int, timeout:int=120) -> np.ndarray:
    """
    Return embedding for text. Currently implemented via Ollama embeddings API.
    """
    if provider == "ollama":

        url = f"{_get_ollama_host()}/api/embeddings"
        resp = requests.post(
            url,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding")
        if not isinstance(vec, list):
            raise ValueError("Invalid embedding response from Ollama")
        return np.asarray(vec, dtype=np.float32)

    # Placeholders for future providers (OpenAI, etc.) can be added here
    raise ValueError("Unsupported EMBED_PROVIDER; currently only 'ollama' is implemented.")

def get_embedding_batch(texts: List[str], provider:str, model:str, dim:int, timeout:int) -> List[List[float]]:
    vectors = [get_embedding(t, provider, model, dim, timeout).tolist() for t in texts]
    return vectors


