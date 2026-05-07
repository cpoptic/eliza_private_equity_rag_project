"""
Dependency injection factory for the RAG pipeline.

Reads EMBEDDER, VECTOR_STORE, and LLM_MODEL env vars to select implementations.
Call build_pipeline() to get a fully wired RAGPipeline instance.
"""

from __future__ import annotations

from src.embedders import get_embedder
from src.llm import get_llm_client
from src.pipeline.prompt_builder import PromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.preprocessing.chunker import SectionAwareChunker
from src.preprocessing.parser import FilingParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_analyzer import RuleBasedQueryAnalyzer
from src.vector_stores import get_vector_store


def build_pipeline() -> RAGPipeline:
    """Assemble and return a fully configured RAGPipeline."""
    embedder = get_embedder()
    store = get_vector_store(dimension=embedder.dimension)
    retriever = HybridRetriever(vector_store=store, embedder=embedder)

    return RAGPipeline(
        parser=FilingParser(),
        chunker=SectionAwareChunker(),
        embedder=embedder,
        vector_store=store,
        retriever=retriever,
        llm_client=get_llm_client(),
        prompt_builder=PromptBuilder(),
        query_analyzer=RuleBasedQueryAnalyzer(),
    )
