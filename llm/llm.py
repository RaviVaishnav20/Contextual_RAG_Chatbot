import os
from dotenv import load_dotenv
from google import genai
from groq import Groq
import requests
import numpy as np
import logging
import json
import boto3

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize clients globally
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
model_id = os.getenv("MODEL_ID")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_SEMANTIC_MODEL", "gemma3:latest")
EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


def generate_content(provider: str, model_name: str, prompt: str) -> str:
    """
    Generates AI content using either Gemini, Groq, or Ollama models based on the provider.

    Args:
        provider: The AI model provider ("gemini", "groq", or "ollama").
        model_name: The specific model name to use (e.g., "gemini-2.5-flash", "llama-3.3-70b-versatile", "llama3").
        prompt: The input text for the AI to process.

    Returns:
        The generated text from the AI.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    if provider.lower() == "bedrock":
        try:
            request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "system": "You are a helpful assistant.",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                }

            response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request)
            )
            response_body = json.loads(response.get("body").read())
            
            return response_body["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Error generating content with Bedrock: {e}")
            raise
    elif provider.lower() == "gemini":
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    elif provider.lower() == "groq":
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
        )
        return chat_completion.choices[0].message.content
    elif provider.lower() == "ollama":
        response = requests.post(OLLAMA_GENERATE_URL, json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=120)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    else:
        raise ValueError("Unsupported provider. Choose 'gemini', 'groq', or 'ollama'.")
    
def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text using local Ollama API"""
    try:
        response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting embedding from Ollama: {e}")
        raise