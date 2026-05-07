"""
LLM client implementation and factory.

Uses LiteLLM for multi-provider support. Select model via LLM_MODEL env var.
Default: claude-sonnet-4-6 (requires ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os

from src.interfaces import BaseLLMClient


class LiteLLMClient(BaseLLMClient):
    """Single-call LLM client wrapping litellm.completion()."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    def complete(self, prompt: str, system: str | None = None) -> str:
        import litellm  # lazy import

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = litellm.completion(model=self._model, messages=messages)
        return response.choices[0].message.content


def get_llm_client() -> BaseLLMClient:
    return LiteLLMClient()
