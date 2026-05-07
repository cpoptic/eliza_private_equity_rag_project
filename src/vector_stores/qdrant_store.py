"""
Qdrant vector store implementation.

Requires a running Qdrant service (Docker or cloud).
Set QDRANT_URL env var (default: http://localhost:6333).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from src.interfaces import BaseVectorStore, Chunk, FilingMetadata, RetrievedChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 256


class QdrantStore(BaseVectorStore):

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "sec_filings",
        dimension: int = 1536,
    ) -> None:
        from qdrant_client import QdrantClient  # lazy import
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._name = collection_name
        self._dimension = dimension

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_chunk_id_to_int(c.chunk_id),
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "ticker": c.metadata.ticker,
                    "company": c.metadata.company,
                    "filing_type": c.metadata.filing_type,
                    "filing_date": c.metadata.filing_date,
                    "report_period": c.metadata.report_period,
                    "fiscal_year": c.metadata.fiscal_year,
                    "quarter": c.metadata.quarter or "",
                    "section": c.section,
                    "section_order": c.section_order,
                    "token_count": c.token_count,
                    "source_file": c.metadata.source_file,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]

        for i in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=self._name,
                points=points[i : i + _UPSERT_BATCH],
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
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        qdrant_filter = _build_filter(filters) if filters else None

        hits = self._client.search(
            collection_name=self._name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                chunk=_payload_to_chunk(hit.payload),
                score=hit.score,
                retrieval_method="dense",
            )
            for hit in hits
        ]

    def get_all_chunks(self) -> list[Chunk]:
        from qdrant_client.models import ScrollRequest

        all_chunks: list[Chunk] = []
        offset = None

        while True:
            results, offset = self._client.scroll(
                collection_name=self._name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_chunks.extend(_payload_to_chunk(r.payload) for r in results)
            if offset is None:
                break

        return all_chunks

    def count(self) -> int:
        info = self._client.get_collection(self._name)
        return info.points_count or 0

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._name)

    def delete_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self._name)
        self._client.create_collection(
            collection_name=self._name,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_int(chunk_id: str) -> int:
    """Convert a string chunk_id to a stable uint64 for Qdrant point IDs."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)


def _build_filter(filters: dict[str, Any]):
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    must: list = []
    for key, value in filters.items():
        if key == "ticker":
            must.append(FieldCondition(key="ticker", match=MatchValue(value=value)))
        elif key == "filing_type":
            must.append(FieldCondition(key="filing_type", match=MatchValue(value=value)))
        elif key == "fiscal_year__gte":
            must.append(FieldCondition(key="fiscal_year", range=Range(gte=value)))
        elif key == "fiscal_year__lte":
            must.append(FieldCondition(key="fiscal_year", range=Range(lte=value)))

    return Filter(must=must) if must else None


def _payload_to_chunk(payload: dict) -> Chunk:
    filing_meta = FilingMetadata(
        company=payload.get("company", ""),
        ticker=payload.get("ticker", ""),
        filing_type=payload.get("filing_type", ""),
        filing_date=payload.get("filing_date", ""),
        report_period=payload.get("report_period", ""),
        quarter=payload.get("quarter") or None,
        cik="",
        source_url="",
        fiscal_year=int(payload.get("fiscal_year", 0)),
        source_file=payload.get("source_file", ""),
    )
    text = payload.get("text", "")
    return Chunk(
        chunk_id=payload.get("chunk_id", ""),
        text=text,
        metadata=filing_meta,
        section=payload.get("section", ""),
        section_order=int(payload.get("section_order", 0)),
        token_count=int(payload.get("token_count", 0)),
        char_start=0,
        char_end=len(text),
    )
