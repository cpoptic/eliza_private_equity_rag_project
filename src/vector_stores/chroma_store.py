"""
ChromaDB vector store implementation.

Uses PersistentClient for local on-disk storage. Translates the abstract
filter dict (ticker, fiscal_year__gte, fiscal_year__lte, filing_type) into
ChromaDB's $where clause syntax.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from src.interfaces import BaseVectorStore, Chunk, FilingMetadata, RetrievedChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 500


class ChromaStore(BaseVectorStore):

    def __init__(self, path: str = "./.chroma", collection_name: str = "sec_filings") -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch_chunks = chunks[i : i + _UPSERT_BATCH]
            batch_embs = embeddings[i : i + _UPSERT_BATCH]
            docs = [c.to_chroma_doc() for c in batch_chunks]
            self._collection.upsert(
                ids=[d["id"] for d in docs],
                documents=[d["document"] for d in docs],
                metadatas=[d["metadata"] for d in docs],
                embeddings=batch_embs,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        where = _build_where(filters) if filters else None
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = _meta_to_chunk(chunk_id, doc, meta)
            # ChromaDB cosine distance → similarity score [0, 1]
            score = max(0.0, 1.0 - dist)
            chunks.append(RetrievedChunk(chunk=chunk, score=score, retrieval_method="dense"))
        return chunks

    def get_all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            _meta_to_chunk(chunk_id, doc, meta)
            for chunk_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._name)
            return True
        except Exception:
            return False

    def delete_collection(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_where(filters: dict[str, Any]) -> dict:
    """Translate abstract filter dict → ChromaDB $where clause."""
    conditions: list[dict] = []
    for key, value in filters.items():
        if key == "ticker":
            conditions.append({"ticker": {"$eq": value}})
        elif key == "fiscal_year__gte":
            conditions.append({"fiscal_year": {"$gte": value}})
        elif key == "fiscal_year__lte":
            conditions.append({"fiscal_year": {"$lte": value}})
        elif key == "filing_type":
            conditions.append({"filing_type": {"$eq": value}})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _meta_to_chunk(chunk_id: str, document: str, meta: dict) -> Chunk:
    """Reconstruct a Chunk from stored ChromaDB id, document, and metadata."""
    filing_meta = FilingMetadata(
        company=meta.get("company", ""),
        ticker=meta.get("ticker", ""),
        filing_type=meta.get("filing_type", ""),
        filing_date=meta.get("filing_date", ""),
        report_period=meta.get("report_period", ""),
        quarter=meta.get("quarter") or None,
        cik="",
        source_url="",
        fiscal_year=int(meta.get("fiscal_year", 0)),
        source_file=meta.get("source_file", ""),
    )
    return Chunk(
        chunk_id=chunk_id,
        text=document,
        metadata=filing_meta,
        section=meta.get("section", ""),
        section_order=int(meta.get("section_order", 0)),
        token_count=int(meta.get("token_count", 0)),
        char_start=0,
        char_end=len(document),
    )
