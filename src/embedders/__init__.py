"""
Embedder implementations and factory.

Select at runtime via EMBEDDER env var:
  "openai"  (default) → OpenAIEmbedder  (text-embedding-3-small, 1536-dim)
  "local"             → LocalEmbedder   (BAAI/bge-large-en-v1.5, 1024-dim)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.interfaces import BaseEmbedder

if TYPE_CHECKING:
    pass

_OPENAI_BATCH = 2048


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small via the openai SDK."""

    def __init__(self, model: str | None = None) -> None:
        import openai  # lazy import so the class is importable without the package
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    @property
    def dimension(self) -> int:
        return 1536

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _OPENAI_BATCH):
            batch = texts[i : i + _OPENAI_BATCH]
            response = self._client.embeddings.create(model=self._model, input=batch)
            results.extend(item.embedding for item in response.data)
        return results


class LocalEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder using BAAI/bge-large-en-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return 1024

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


def get_embedder() -> BaseEmbedder:
    """Factory: reads EMBEDDER env var to select implementation."""
    choice = os.getenv("EMBEDDER", "openai").lower()
    if choice == "local":
        return LocalEmbedder()
    return OpenAIEmbedder()
