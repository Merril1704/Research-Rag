"""
LLM Client — Abstraction over Groq API with Ollama fallback.

Provides a single generate() interface. Tries Groq first;
if it fails, falls back to local Ollama.

Usage:
    from researchrag.generation.llm_client import LLMClient

    client = LLMClient()
    response = client.generate("Your prompt here")
"""

import requests

from groq import Groq

from researchrag.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_FALLBACK_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from researchrag.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    LLM client with Groq as primary and Ollama as fallback.

    Attributes:
        groq_client: Groq SDK client instance.
        model: Primary Groq model name.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Groq client
        self.groq_client = None
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            logger.info(f"Groq client initialized (model: {model})")
        else:
            logger.warning(
                "No GROQ_API_KEY found. Will use Ollama fallback only."
            )

    def generate(self, prompt: str, model: str | None = None) -> str:
        """
        Generate text from a prompt.

        Tries Groq first, falls back to Ollama if Groq fails.

        Args:
            prompt: The full prompt string.
            model: Override model name (optional).

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If both Groq and Ollama fail.
        """
        use_model = model or self.model

        # Try Groq first
        if self.groq_client:
            try:
                return self._generate_groq(prompt, use_model)
            except Exception as e:
                logger.warning(
                    f"Groq generation failed ({use_model}): {e}. "
                    f"Trying fallback model..."
                )
                # Try fallback Groq model
                if use_model != GROQ_FALLBACK_MODEL:
                    try:
                        return self._generate_groq(prompt, GROQ_FALLBACK_MODEL)
                    except Exception as e2:
                        logger.warning(
                            f"Groq fallback also failed: {e2}. "
                            f"Trying Ollama..."
                        )

        # Fallback to Ollama
        try:
            return self._generate_ollama(prompt)
        except Exception as e:
            logger.error(f"All LLM backends failed: {e}", exc_info=True)
            raise RuntimeError(
                "Generation failed. Neither Groq nor Ollama responded. "
                "Check your API key or ensure Ollama is running locally."
            ) from e

    def _generate_groq(self, prompt: str, model: str) -> str:
        """Generate via Groq API."""
        logger.info(f"Generating via Groq (model={model})")
        logger.debug(f"Prompt length: {len(prompt)} chars")

        response = self.groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = response.choices[0].message.content
        logger.info(f"Groq response: {len(text)} chars generated")
        logger.debug(
            f"Usage: prompt_tokens={response.usage.prompt_tokens}, "
            f"completion_tokens={response.usage.completion_tokens}"
        )
        return text

    def _generate_ollama(self, prompt: str) -> str:
        """Generate via local Ollama API."""
        logger.info(
            f"Generating via Ollama (model={OLLAMA_MODEL}, "
            f"url={OLLAMA_BASE_URL})"
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "")
        logger.info(f"Ollama response: {len(text)} chars generated")
        return text
