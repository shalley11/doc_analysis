"""LLM Generator module for answer generation with citations."""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate a response from the LLM."""
        pass


class OllamaGenerator(LLMGenerator):
    """Generator using Ollama for local LLM inference."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


class OpenAIGenerator(LLMGenerator):
    """Generator using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")


class AnthropicGenerator(LLMGenerator):
    """Generator using Anthropic API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")


class GeminiGenerator(LLMGenerator):
    """Generator using Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        # Combine system and user prompts for Gemini
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        payload = {
            "contents": [{"parts": [{"text": combined_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        try:
            response = requests.post(
                url, headers=headers, params=params, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")


def create_generator(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMGenerator:
    """
    Factory function to create an LLM generator.

    Args:
        provider: LLM provider ("ollama", "openai", "anthropic", "gemini")
        model: Model name (uses defaults if not specified)
        api_key: API key for cloud providers
        base_url: Base URL for Ollama

    Returns:
        LLMGenerator instance
    """
    provider = provider.lower()

    if provider == "ollama":
        return OllamaGenerator(
            model=model or "llama3",
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    elif provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required")
        return OpenAIGenerator(api_key=key, model=model or "gpt-4o-mini")

    elif provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required")
        return AnthropicGenerator(api_key=key, model=model or "claude-3-haiku-20240307")

    elif provider == "gemini":
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API key required")
        return GeminiGenerator(api_key=key, model=model or "gemini-1.5-flash")

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_generator() -> LLMGenerator:
    """
    Get the default generator based on available API keys.

    Priority: Ollama > OpenAI > Anthropic > Gemini
    """
    # Check for Ollama first (local, no API key needed)
    try:
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return create_generator("ollama")
    except:
        pass

    # Check for cloud API keys
    if os.environ.get("OPENAI_API_KEY"):
        return create_generator("openai")

    if os.environ.get("ANTHROPIC_API_KEY"):
        return create_generator("anthropic")

    if os.environ.get("GOOGLE_API_KEY"):
        return create_generator("gemini")

    # Fallback to Ollama (will fail if not available)
    return create_generator("ollama")
