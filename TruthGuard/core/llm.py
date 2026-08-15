# llm.py - LLM provider abstraction

import os
from typing import Optional, Any
from openai import OpenAI
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using OpenAI-compatible API."""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key)
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate a response using OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenRouter generation failed: {str(e)}")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key)
        self.model = model
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate a response using Google Gemini."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                **kwargs
            )
            response = self.model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Google Gemini generation failed: {str(e)}")


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key)
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate a response using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")


def get_llm_provider(provider: str, model: str, api_key: str) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    providers = {
        "openrouter": OpenRouterProvider,
        "google": GoogleProvider,
        "openai": OpenAIProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider](api_key, model)


def create_base_provider(settings: Any) -> LLMProvider:
    """Create the base model provider from settings."""
    api_key_map = {
        "openrouter": settings.openrouter_api_key,
        "google": settings.google_api_key,
        "openai": settings.openai_api_key
    }
    
    api_key = api_key_map.get(settings.base_provider)
    if not api_key:
        raise ValueError(f"API key not found for provider: {settings.base_provider}")
    
    return get_llm_provider(settings.base_provider, settings.base_model, api_key)


def create_judge_provider(settings: Any) -> LLMProvider:
    """Create the judge model provider from settings."""
    api_key_map = {
        "openrouter": settings.openrouter_api_key,
        "google": settings.google_api_key,
        "openai": settings.openai_api_key
    }
    
    api_key = api_key_map.get(settings.judge_provider)
    if not api_key:
        raise ValueError(f"API key not found for provider: {settings.judge_provider}")
    
    return get_llm_provider(settings.judge_provider, settings.judge_model, api_key)
