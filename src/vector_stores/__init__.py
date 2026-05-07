"""
Vector store factory.

Select at runtime via VECTOR_STORE env var:
  "chroma"  (default) → ChromaStore  (local persistent ChromaDB)
  "qdrant"            → QdrantStore  (requires running Qdrant service)
"""

from __future__ import annotations

import os

from src.interfaces import BaseVectorStore


def get_vector_store(dimension: int = 1536) -> BaseVectorStore:
    """Factory: reads VECTOR_STORE env var to select implementation."""
    choice = os.getenv("VECTOR_STORE", "chroma").lower()

    if choice == "qdrant":
        from src.vector_stores.qdrant_store import QdrantStore
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = os.getenv("QDRANT_COLLECTION", "sec_filings")
        return QdrantStore(url=url, collection_name=collection, dimension=dimension)

    # default: chroma
    from src.vector_stores.chroma_store import ChromaStore
    path = os.getenv("CHROMA_PATH", "./.chroma")
    collection = os.getenv("CHROMA_COLLECTION", "sec_filings")
    return ChromaStore(path=path, collection_name=collection)
