"""
LLM Client — Model-agnostic interface for text generation.
Supports: Anthropic (Claude), OpenAI-compatible (vLLM, LM Studio), Ollama.
Switch provider in settings.yaml without code changes.
"""

import os
from typing import Generator, Optional

from loguru import logger


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(
        self,
        provider: str = "anthropic",
        config: dict = None,
    ):
        self.provider = provider
        self.config = config or {}
        self._client = None

        logger.info(f"LLMClient: provider={provider}")

    def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in .env")
            self._client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "openai_compatible":
            from openai import OpenAI
            base_url = self.config.get("base_url", "http://localhost:8000/v1")
            api_key = os.getenv("OPENAI_API_KEY", "not-needed")
            self._client = OpenAI(base_url=base_url, api_key=api_key)

        elif self.provider == "ollama":
            from openai import OpenAI
            base_url = self.config.get("base_url", "http://localhost:11434") + "/v1"
            self._client = OpenAI(base_url=base_url, api_key="ollama")

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        return self._client

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instructions
            user_message: User query with context
            max_tokens: Override config max_tokens
            temperature: Override config temperature

        Returns:
            Generated text response
        """
        client = self._get_client()
        tokens = max_tokens or self.config.get("max_tokens", 4096)
        temp = temperature if temperature is not None else self.config.get("temperature", 0.2)
        model = self.config.get("model", "")

        try:
            if self.provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=tokens,
                    temperature=temp,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                result = response.content[0].text
                logger.info(
                    f"Claude response: {len(result)} chars, "
                    f"tokens: {response.usage.input_tokens}in/{response.usage.output_tokens}out"
                )
                return result

            else:  # OpenAI-compatible / Ollama
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=tokens,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                result = response.choices[0].message.content
                logger.info(f"LLM response: {len(result)} chars")
                return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Stream a response token by token."""
        client = self._get_client()
        tokens = max_tokens or self.config.get("max_tokens", 4096)
        temp = temperature if temperature is not None else self.config.get("temperature", 0.2)
        model = self.config.get("model", "")

        try:
            if self.provider == "anthropic":
                with client.messages.stream(
                    model=model,
                    max_tokens=tokens,
                    temperature=temp,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ) as stream:
                    for text in stream.text_stream:
                        yield text

            else:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=tokens,
                    temperature=temp,
                    stream=True,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise
