import os
import json
import logging
from typing import List

import numpy as np
import requests

from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.application.networks.embeddings import EmbeddingModelSingleton
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
        # model_id = os.getenv("MODEL_ID")
        resp = client.invoke_model(modelId=model_name, body=json.dumps(request))
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
    if prov == "hugging_face":
        
        _model = TextGenerationModelSingleton()
        response = _model(prompt)
        return response["content"]

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
    elif provider =="hugging_face":
        _model=EmbeddingModelSingleton()
        return _model(text)

    # Placeholders for future providers (OpenAI, etc.) can be added here
    raise ValueError("Unsupported EMBED_PROVIDER; currently only 'ollama' is implemented.")

# def get_embedding_batch(texts: List[str], provider:str, model:str, dim:int, timeout:int) -> List[List[float]]:
#     vectors = [get_embedding(t, provider, model, dim, timeout).tolist() for t in texts]
#     return vectors
    
def get_embedding_batch(texts: List[str], provider:str, model:str, dim:int, timeout:int) -> List[List[float]]:
    vectors = [get_embedding(t, provider, model, dim, timeout) for t in texts]
    return vectors


from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from enum import Enum

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoModelForImageTextToText
)
from contextual_rag.settings import settings

from .base import SingletonMeta


class ModelProvider(Enum):
    """Supported model providers"""
    QWEN = "qwen"
    GEMMA = "gemma"


class TextGenerationModelSingleton(metaclass=SingletonMeta):
    """
    A singleton class that provides text generation capabilities for different model providers.
    Supports Qwen models with thinking mode and Gemma models with multimodal capabilities.
    """

    def __init__(
        self,
        model_id: str = settings.TEXT_GENERATION_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._cache_dir = cache_dir
        
        # Determine provider based on model name
        self._provider = self._detect_provider(model_id)
        
        # Initialize model and tokenizer/processor based on provider
        self._initialize_model()

    def _detect_provider(self, model_id: str) -> ModelProvider:
        """Detect the provider based on model ID"""
        model_lower = model_id.lower()
        if "qwen" in model_lower:
            return ModelProvider.QWEN
        elif "gemma" in model_lower:
            return ModelProvider.GEMMA
        else:
            logger.warning(f"Unknown provider for {model_id}, defaulting to QWEN")
            return ModelProvider.QWEN

    def _initialize_model(self) -> None:
        """Initialize model and tokenizer/processor based on provider"""
        try:
            if self._provider == ModelProvider.QWEN:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    dtype="auto",
                    device_map="auto",
                    cache_dir=str(self._cache_dir) if self._cache_dir else None
                )
                
            elif self._provider == ModelProvider.GEMMA:
                self._processor = AutoProcessor.from_pretrained(
                    self._model_id,
                    cache_dir=str(self._cache_dir) if self._cache_dir else None
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id,
                    dtype="auto",
                    device_map="auto",
                    cache_dir=str(self._cache_dir) if self._cache_dir else None
                )
                
            self._model.eval()
            logger.info(f"Successfully loaded {self._provider.value} model: {self._model_id}")
            
        except Exception as e:
            logger.error(f"Error loading model {self._model_id}: {str(e)}")
            raise

    @property
    def model_id(self) -> str:
        """Returns the model identifier"""
        return self._model_id

    @property
    def provider(self) -> ModelProvider:
        """Returns the detected provider"""
        return self._provider

    @cached_property
    def max_context_length(self) -> int:
        """Returns the maximum context length for the model"""
        if self._provider == ModelProvider.QWEN:
            return getattr(self._tokenizer, 'model_max_length', 32768)
        elif self._provider == ModelProvider.GEMMA:
            return getattr(self._processor.tokenizer, 'model_max_length', 8192)
        return 2048

    def _generate_qwen_response(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        **kwargs
    ) -> Dict[str, str]:
        """Generate response using Qwen model with thinking capability"""
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Tokenize input
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get('do_sample', True),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Extract output tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if enabled
        thinking_content = ""
        content = ""
        
        if enable_thinking:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self._tokenizer.decode(
                    output_ids[:index], 
                    skip_special_tokens=True
                ).strip("\n")
                content = self._tokenizer.decode(
                    output_ids[index:], 
                    skip_special_tokens=True
                ).strip("\n")
            except ValueError:
                # No thinking tokens found
                content = self._tokenizer.decode(
                    output_ids, 
                    skip_special_tokens=True
                ).strip("\n")
        else:
            content = self._tokenizer.decode(
                output_ids, 
                skip_special_tokens=True
            ).strip("\n")
        
        return {
            "content": content,
            "thinking_content": thinking_content if enable_thinking else None
        }

    def _generate_gemma_response(
        self, 
        messages: List[Dict[str, Any]], 
        max_new_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, str]:
        """Generate response using Gemma model with multimodal capability"""
        
        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get('do_sample', True),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
            )
        
        # Decode response
        content = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        return {
            "content": content,
            "thinking_content": None
        }

    def __call__(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate text response based on input messages.
        
        Args:
            messages: Either a string prompt or list of message dicts
            max_new_tokens: Maximum number of tokens to generate
            enable_thinking: Enable thinking mode for Qwen models (ignored for Gemma)
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing 'content' and 'thinking_content' (if applicable)
        """
        
        # Convert string input to message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        try:
            if self._provider == ModelProvider.QWEN:
                return self._generate_qwen_response(
                    messages, 
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    **kwargs
                )
            elif self._provider == ModelProvider.GEMMA:
                return self._generate_gemma_response(
                    messages, 
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Error generating response with {self._model_id}: {str(e)}")
            return {
                "content": "",
                "thinking_content": None
            }

    def generate_simple(self, prompt: str, **kwargs) -> str:
        """
        Simple text generation method that returns only the content.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text content
        """
        result = self(prompt, **kwargs)
        return result["content"]