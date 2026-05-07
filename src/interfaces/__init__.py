"""
Abstract base classes for all swappable RAG components.

Design principle: every concrete implementation lives in its own module and
imports ONLY from this file. The pipeline assembles implementations at
runtime via config — never by importing concrete classes directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared data models
# ---------------------------------------------------------------------------

@dataclass
class FilingMetadata:
    """Structured metadata parsed from the 10-line header block of each file."""
    company: str
    ticker: str
    filing_type: str          # "10-K" | "10-Q"
    filing_date: str          # ISO date string
    report_period: str        # ISO date string
    quarter: str | None       # e.g. "2024Q3", None for 10-Ks
    cik: str
    source_url: str
    fiscal_year: int          # derived from filing_date
    source_file: str          # original filename


@dataclass
class Chunk:
    """
    A single unit of indexed text. Carries all metadata needed for
    filtered retrieval and LLM citation formatting.
    """
    chunk_id: str             # deterministic: {ticker}_{filing_type}_{period}_{section}_{idx}
    text: str
    metadata: FilingMetadata
    section: str              # e.g. "Item 1A - Risk Factors"
    section_order: int        # position in filing for ranking tie-breaking
    token_count: int
    char_start: int           # offset in original cleaned text
    char_end: int

    def provenance_header(self) -> str:
        """Human-readable header injected above each chunk in the LLM prompt."""
        period = self.metadata.report_period[:7]  # YYYY-MM
        return (
            f"[{self.metadata.ticker} | {self.metadata.filing_type} | "
            f"{period} | {self.section}]"
        )

    def to_chroma_doc(self) -> dict:
        """Serialize to ChromaDB add() format."""
        return {
            "id": self.chunk_id,
            "document": self.text,
            "metadata": {
                "ticker": self.metadata.ticker,
                "company": self.metadata.company,
                "filing_type": self.metadata.filing_type,
                "filing_date": self.metadata.filing_date,
                "report_period": self.metadata.report_period,
                "fiscal_year": self.metadata.fiscal_year,
                "quarter": self.metadata.quarter or "",
                "section": self.section,
                "section_order": self.section_order,
                "token_count": self.token_count,
                "source_file": self.metadata.source_file,
            },
        }


@dataclass
class RetrievedChunk:
    """A chunk plus its retrieval score, returned from any retriever."""
    chunk: Chunk
    score: float
    retrieval_method: str     # "dense" | "bm25" | "hybrid"


@dataclass
class QueryContext:
    """
    Parsed intent extracted from the user's natural-language question.
    Built by QueryAnalyzer; consumed by retrievers.
    """
    original_query: str
    tickers: list[str] = field(default_factory=list)
    year_range: tuple[int, int] | None = None   # (start_year, end_year)
    section_hints: list[str] = field(default_factory=list)  # ["Item 1A", "Item 7"]
    query_type: str = "general"   # "comparison" | "trend" | "thematic" | "general"
    sub_queries: list[str] = field(default_factory=list)  # one per ticker for comparisons


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Splits a raw filing text into a list of Chunks."""

    @abstractmethod
    def chunk(self, text: str, metadata: FilingMetadata) -> list[Chunk]:
        """
        Parse and chunk a single filing's cleaned text.

        Args:
            text: Cleaned filing text (XBRL stripped, header removed).
            metadata: Parsed header metadata for this filing.

        Returns:
            Ordered list of Chunk objects.
        """
        ...


class BaseEmbedder(ABC):
    """Generates dense vector embeddings for text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension. Must be consistent across calls."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, same length as input.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Convenience wrapper for single query embedding."""
        return self.embed_texts([query])[0]


class BaseVectorStore(ABC):
    """Stores and retrieves chunk embeddings with metadata filtering."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with their precomputed embeddings."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Approximate nearest-neighbour search with optional metadata filters.

        Args:
            query_embedding: Dense query vector.
            top_k: Number of results to return.
            filters: Key/value metadata filters (e.g. {"ticker": "AAPL"}).

        Returns:
            List of RetrievedChunks ordered by descending similarity.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed chunks."""
        ...

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        """Return all chunks (used to build BM25 index from stored data)."""
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        """Check whether the collection/index has been built."""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop and recreate collection (for re-indexing)."""
        ...


class BaseRetriever(ABC):
    """
    Retrieves relevant chunks for a QueryContext.

    Implementations can be dense-only, BM25-only, or hybrid.
    Each implementation is responsible for sub-query decomposition.
    """

    @abstractmethod
    def retrieve(self, context: QueryContext, top_k: int) -> list[RetrievedChunk]:
        """
        Run retrieval for a parsed query context.

        Args:
            context: Parsed query intent including tickers, date range, etc.
            top_k: Final number of chunks to return after merging/reranking.

        Returns:
            Ranked list of RetrievedChunks ready for prompt assembly.
        """
        ...


class BaseQueryAnalyzer(ABC):
    """Parses a natural-language question into a QueryContext."""

    @abstractmethod
    def analyze(self, question: str) -> QueryContext:
        """
        Extract structured intent from a raw question string.

        Must be pure rule-based — no LLM calls (those are reserved
        for the single final answer call).
        """
        ...


class BaseLLMClient(ABC):
    """Wraps the single LLM API call that produces the final answer."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """
        Make exactly one LLM API call and return the response text.

        This is the ONLY LLM call in the pipeline.
        """
        ...
