"""
Hybrid retriever: dense vector search + BM25, merged with Reciprocal Rank Fusion.

Design:
  - For comparison queries (multiple tickers): run independent per-ticker
    sub-queries and merge results, ensuring balanced coverage per company.
  - For all queries: combine dense cosine similarity with BM25 exact-match
    scores via RRF to catch both semantic and terminological matches.
  - Final reranking respects section_order for tie-breaking within a company.

RRF formula:  score(d) = Σ 1 / (k + rank(d))   where k=60 (standard).
"""

from __future__ import annotations

import os
from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.interfaces import (
    BaseEmbedder,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    QueryContext,
    RetrievedChunk,
)

_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Dense + BM25 hybrid retriever with RRF fusion.

    Args:
        vector_store: Any BaseVectorStore implementation.
        embedder: Any BaseEmbedder implementation.
        top_k_dense: Candidates to fetch per dense sub-query.
        top_k_bm25: Candidates to fetch from BM25.
        top_k_final: Final chunks returned after fusion.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
        top_k_final: int | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._top_k_dense = int(top_k_dense or os.getenv("TOP_K_DENSE", 20))
        self._top_k_bm25 = int(top_k_bm25 or os.getenv("TOP_K_BM25", 20))
        self._top_k_final = int(top_k_final or os.getenv("TOP_K_FINAL", 12))

        # BM25 index is built lazily on first retrieval call
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, context: QueryContext, top_k: int | None = None) -> list[RetrievedChunk]:
        final_k = top_k or self._top_k_final
        self._ensure_bm25_index()

        if context.query_type == "comparison" and len(context.tickers) > 1:
            return self._retrieve_comparison(context, final_k)
        else:
            return self._retrieve_single(context.original_query, context, final_k)

    # ------------------------------------------------------------------
    # Retrieval paths
    # ------------------------------------------------------------------

    def _retrieve_comparison(
        self, context: QueryContext, final_k: int
    ) -> list[RetrievedChunk]:
        """
        Per-ticker retrieval for comparison queries.

        Allocates final_k slots proportionally across tickers, then merges.
        E.g. 3 companies, final_k=12 → ~4 chunks per company.
        """
        n = len(context.tickers)
        per_ticker_k = max(2, final_k // n)

        all_chunks: list[RetrievedChunk] = []
        for ticker, sub_query in zip(context.tickers, context.sub_queries):
            filters = self._build_filters(context, ticker_override=ticker)
            chunks = self._retrieve_single(sub_query, context, per_ticker_k, filters)
            all_chunks.extend(chunks)

        # If any ticker returned nothing, fill from unfiltered search
        tickers_found = {r.chunk.metadata.ticker for r in all_chunks}
        for ticker in context.tickers:
            if ticker not in tickers_found:
                sub_q = next(
                    (q for q, t in zip(context.sub_queries, context.tickers) if t == ticker),
                    context.original_query,
                )
                fallback = self._retrieve_single(sub_q, context, per_ticker_k)
                for r in fallback:
                    if r.chunk.metadata.ticker == ticker:
                        all_chunks.append(r)

        # Deduplicate by chunk_id, keep highest score
        return _dedup_and_limit(all_chunks, final_k)

    def _retrieve_single(
        self,
        query_text: str,
        context: QueryContext,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + BM25 retrieval for a single query string."""
        if filters is None:
            filters = self._build_filters(context)

        # Dense retrieval
        query_emb = self._embedder.embed_query(query_text)
        dense_results = self._store.query(
            query_embedding=query_emb,
            top_k=self._top_k_dense,
            filters=filters if filters else None,
        )

        # BM25 retrieval (filtered post-hoc)
        bm25_results = self._bm25_search(query_text, top_k=self._top_k_bm25, filters=filters)

        # Fuse with RRF
        fused = _reciprocal_rank_fusion(dense_results, bm25_results, k=_RRF_K)

        # Apply section preference if section hints given
        if context.section_hints:
            fused = _boost_section_matches(fused, context.section_hints)

        return fused[:top_k]

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _ensure_bm25_index(self) -> None:
        if self._bm25 is not None:
            return
        self._bm25_chunks = self._store.get_all_chunks()
        tokenized = [_tokenize(c.text) for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with chunks and apply metadata filters
        scored: list[tuple[float, Chunk]] = []
        for score, chunk in zip(scores, self._bm25_chunks):
            if score <= 0:
                continue
            if filters and not _chunk_matches_filters(chunk, filters):
                continue
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        max_score = top[0][0] if top else 1.0

        return [
            RetrievedChunk(
                chunk=chunk,
                score=score / max_score,  # normalise to [0,1]
                retrieval_method="bm25",
            )
            for score, chunk in top
        ]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        context: QueryContext,
        ticker_override: str | None = None,
    ) -> dict:
        filters: dict = {}

        ticker = ticker_override
        if not ticker and len(context.tickers) == 1:
            ticker = context.tickers[0]
        if ticker:
            filters["ticker"] = ticker

        if context.year_range:
            start, end = context.year_range
            filters["fiscal_year__gte"] = start
            filters["fiscal_year__lte"] = end

        return filters


# ---------------------------------------------------------------------------
# RRF and helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    bm25: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, result in enumerate(dense):
        cid = result.chunk.chunk_id
        scores[cid] += 1.0 / (k + rank + 1)
        chunk_map[cid] = RetrievedChunk(
            chunk=result.chunk, score=0.0, retrieval_method="hybrid"
        )

    for rank, result in enumerate(bm25):
        cid = result.chunk.chunk_id
        scores[cid] += 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = RetrievedChunk(
                chunk=result.chunk, score=0.0, retrieval_method="hybrid"
            )

    # Write final RRF scores back
    for cid, score in scores.items():
        chunk_map[cid].score = score

    return sorted(chunk_map.values(), key=lambda r: r.score, reverse=True)


def _boost_section_matches(
    results: list[RetrievedChunk],
    section_hints: list[str],
) -> list[RetrievedChunk]:
    """Multiply score by 1.3 for chunks whose section matches a hint."""
    boosted = []
    for r in results:
        multiplier = 1.0
        for hint in section_hints:
            hint_key = hint.split(" - ")[0].lower()  # "item 1a"
            if hint_key in r.chunk.section.lower():
                multiplier = 1.3
                break
        boosted.append(
            RetrievedChunk(
                chunk=r.chunk,
                score=r.score * multiplier,
                retrieval_method=r.retrieval_method,
            )
        )
    return sorted(boosted, key=lambda r: r.score, reverse=True)


def _dedup_and_limit(
    results: list[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        if r.chunk.chunk_id not in seen:
            seen.add(r.chunk.chunk_id)
            out.append(r)
        if len(out) >= limit:
            break
    return out


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _chunk_matches_filters(chunk: Chunk, filters: dict) -> bool:
    """Post-hoc filter check for BM25 results."""
    meta = chunk.metadata
    for key, value in filters.items():
        if key == "ticker":
            if meta.ticker != value:
                return False
        elif key == "fiscal_year__gte":
            if meta.fiscal_year < value:
                return False
        elif key == "fiscal_year__lte":
            if meta.fiscal_year > value:
                return False
        elif key == "filing_type":
            if meta.filing_type != value:
                return False
    return True
