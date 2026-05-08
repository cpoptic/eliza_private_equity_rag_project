"""
Embedder implementations and factory.

Select at runtime via EMBEDDER env var:
  "openai"  (default) → OpenAIEmbedder  (text-embedding-3-small, 1536-dim)
  "local"             → LocalEmbedder   (BAAI/bge-large-en-v1.5, 1024-dim)
"""

from __future__ import annotations

import logging
import os
import re

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)

_OPENAI_BATCH = 512   # texts per API call (smaller batches → easier rate control)

# Pattern to extract the batch index from OpenAI's token-limit error messages.
# Example: "Invalid 'input[13]': maximum input length is 8192 tokens."
_INPUT_IDX_RE = re.compile(r"input\[(\d+)\]")


def _openai_retry_decorator():
    """Tenacity retry: exponential backoff on RateLimitError, up to 8 attempts."""
    import openai
    return retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_random_exponential(min=1, max=90),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small with rate-limit retry via tenacity."""

    def __init__(self, model: str | None = None) -> None:
        import openai
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = openai.OpenAI()

    @property
    def dimension(self) -> int:
        return 1536

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _OPENAI_BATCH):
            batch = texts[i : i + _OPENAI_BATCH]
            response = self._call_api(batch)
            results.extend(item.embedding for item in response.data)
        return results

    def embed_texts_with_ids(
        self, texts: list[str], ids: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """
        Embed texts and return (embeddings, failed_ids).

        On token-limit errors, retries the batch item-by-item to identify
        which specific chunk is oversized and logs its ID instead of failing
        the whole batch.
        """
        import openai

        embeddings: list[list[float]] = []
        failed_ids: list[str] = []

        for i in range(0, len(texts), _OPENAI_BATCH):
            batch_texts = texts[i : i + _OPENAI_BATCH]
            batch_ids = ids[i : i + _OPENAI_BATCH]
            try:
                response = self._call_api(batch_texts)
                embeddings.extend(item.embedding for item in response.data)
            except openai.BadRequestError as exc:
                # Token-limit error: fall back to one-at-a-time to identify culprit
                logger.warning(
                    "Batch [%d:%d] failed (%s); falling back to per-item embedding",
                    i, i + len(batch_texts), exc,
                )
                for text, chunk_id in zip(batch_texts, batch_ids):
                    try:
                        response = self._call_api([text])
                        embeddings.append(response.data[0].embedding)
                    except openai.BadRequestError as item_exc:
                        logger.error(
                            "Skipping chunk %s — exceeds token limit: %s",
                            chunk_id, item_exc,
                        )
                        failed_ids.append(chunk_id)

        return embeddings, failed_ids

    @_openai_retry_decorator()
    def _call_api(self, batch: list[str]):
        return self._client.embeddings.create(model=self._model, input=batch)


class LocalEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder using BAAI/bge-large-en-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return 1024

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_texts_with_ids(
        self, texts: list[str], ids: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        return self.embed_texts(texts), []


def get_embedder() -> BaseEmbedder:
    """Factory: reads EMBEDDER env var to select implementation."""
    choice = os.getenv("EMBEDDER", "openai").lower()
    if choice == "local":
        return LocalEmbedder()
    return OpenAIEmbedder()
