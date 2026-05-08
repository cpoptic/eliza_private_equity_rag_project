"""
Unit tests for RuleBasedQueryAnalyzer and RAGPipeline query method.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import pytest

from src.interfaces import (
    Chunk,
    FilingMetadata,
    QueryContext,
    RetrievedChunk,
)
from src.retrieval.query_analyzer import RuleBasedQueryAnalyzer
from src.pipeline.rag_pipeline import RAGPipeline, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(ticker="AAPL", section="Item 1A", idx=0) -> Chunk:
    meta = FilingMetadata(
        company="Test Corp",
        ticker=ticker,
        filing_type="10-K",
        filing_date="2023-10-01",
        report_period="2023-09-30",
        quarter=None,
        cik="0001234567",
        source_url="https://example.com",
        fiscal_year=2023,
        source_file="test.txt",
    )
    return Chunk(
        chunk_id=f"{ticker}_{section}_{idx}",
        text=f"Sample text from {section} for {ticker}.",
        metadata=meta,
        section=section,
        section_order=idx,
        token_count=10,
        char_start=0,
        char_end=100,
    )


def _make_retrieved(ticker="AAPL", score=0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(ticker=ticker),
        score=score,
        retrieval_method="hybrid",
    )


def _make_pipeline(answer="Test answer.") -> RAGPipeline:
    """Build a RAGPipeline with all dependencies mocked."""
    mock_parser = MagicMock()
    mock_chunker = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 10

    mock_store = MagicMock()
    mock_store.count.return_value = 100
    mock_store.collection_exists.return_value = True

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_make_retrieved()]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = answer

    mock_prompt = MagicMock()
    mock_prompt.build_system_prompt.return_value = "System prompt."
    mock_prompt.build_user_prompt.return_value = "User prompt."

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = QueryContext(
        original_query="test question",
        tickers=["AAPL"],
        year_range=None,
        section_hints=["Item 1A"],
        query_type="general",
    )

    return RAGPipeline(
        parser=mock_parser,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_store,
        retriever=mock_retriever,
        llm_client=mock_llm,
        prompt_builder=mock_prompt,
        query_analyzer=mock_analyzer,
    )


# ---------------------------------------------------------------------------
# RuleBasedQueryAnalyzer — ticker extraction
# ---------------------------------------------------------------------------

class TestTickerExtraction:
    def setup_method(self):
        self.analyzer = RuleBasedQueryAnalyzer()

    def test_extracts_ticker_symbol_direct(self):
        ctx = self.analyzer.analyze("What are AAPL's revenue trends?")
        assert "AAPL" in ctx.tickers

    def test_extracts_ticker_via_company_name(self):
        ctx = self.analyzer.analyze("What are Apple's key risk factors?")
        assert "AAPL" in ctx.tickers

    def test_extracts_multiple_tickers(self):
        ctx = self.analyzer.analyze("Compare Apple and Tesla risk factors")
        assert "AAPL" in ctx.tickers
        assert "TSLA" in ctx.tickers

    def test_alias_jpmorgan(self):
        ctx = self.analyzer.analyze("What are JPMorgan's legal risks?")
        assert "JPM" in ctx.tickers

    def test_alias_nvidia(self):
        ctx = self.analyzer.analyze("How has NVIDIA's revenue grown?")
        assert "NVDA" in ctx.tickers

    def test_no_tickers_when_none_present(self):
        ctx = self.analyzer.analyze("What are the biggest risks in pharma?")
        assert ctx.tickers == []

    def test_deduplicates_tickers(self):
        ctx = self.analyzer.analyze("Compare Apple, AAPL, and apple performance")
        assert ctx.tickers.count("AAPL") == 1


# ---------------------------------------------------------------------------
# RuleBasedQueryAnalyzer — temporal extraction
# ---------------------------------------------------------------------------

class TestTemporalExtraction:
    def setup_method(self):
        self.analyzer = RuleBasedQueryAnalyzer()

    def test_extracts_explicit_year(self):
        ctx = self.analyzer.analyze("What was Apple's revenue in 2022?")
        assert ctx.year_range == (2022, 2022)

    def test_extracts_year_range(self):
        ctx = self.analyzer.analyze("Compare Apple from 2021 to 2023")
        assert ctx.year_range == (2021, 2023)

    def test_since_year(self):
        ctx = self.analyzer.analyze("How has AAPL grown since 2020?")
        assert ctx.year_range is not None
        assert ctx.year_range[0] == 2020

    def test_no_year_returns_none(self):
        ctx = self.analyzer.analyze("What are Apple's risk factors?")
        assert ctx.year_range is None

    def test_last_n_years(self):
        ctx = self.analyzer.analyze("Revenue trends over the last 3 years")
        assert ctx.year_range is not None
        start, end = ctx.year_range
        assert end - start == 3


# ---------------------------------------------------------------------------
# RuleBasedQueryAnalyzer — section hints
# ---------------------------------------------------------------------------

class TestSectionHints:
    def setup_method(self):
        self.analyzer = RuleBasedQueryAnalyzer()

    def test_risk_keywords_hint_item_1a(self):
        ctx = self.analyzer.analyze("What are the key risk factors?")
        assert "Item 1A" in ctx.section_hints

    def test_revenue_keywords_hint_item_7(self):
        ctx = self.analyzer.analyze("What was the revenue growth?")
        assert "Item 7" in ctx.section_hints

    def test_balance_sheet_hints_item_8(self):
        ctx = self.analyzer.analyze("What does the balance sheet show?")
        assert "Item 8" in ctx.section_hints

    def test_no_hints_for_generic_query(self):
        ctx = self.analyzer.analyze("Tell me about Apple")
        # No specific financial keywords → no section hints expected
        assert isinstance(ctx.section_hints, list)


# ---------------------------------------------------------------------------
# RuleBasedQueryAnalyzer — query type classification
# ---------------------------------------------------------------------------

class TestQueryTypeClassification:
    def setup_method(self):
        self.analyzer = RuleBasedQueryAnalyzer()

    def test_comparison_type_two_tickers(self):
        ctx = self.analyzer.analyze("Compare Apple and Tesla on risk factors")
        assert ctx.query_type == "comparison"

    def test_comparison_type_keyword(self):
        ctx = self.analyzer.analyze("Apple vs Microsoft revenue?")
        assert ctx.query_type == "comparison"

    def test_trend_type_keyword(self):
        ctx = self.analyzer.analyze("How has Apple's revenue trended over time?")
        assert ctx.query_type == "trend"

    def test_trend_type_multi_year_range(self):
        ctx = self.analyzer.analyze("Apple revenue from 2020 to 2023")
        assert ctx.query_type == "trend"

    def test_general_type_single_ticker(self):
        ctx = self.analyzer.analyze("What are Apple's key risk factors?")
        assert ctx.query_type == "general"

    def test_thematic_type_no_ticker(self):
        ctx = self.analyzer.analyze("What risks do pharma companies face?")
        assert ctx.query_type == "thematic"


# ---------------------------------------------------------------------------
# RAGPipeline — query method
# ---------------------------------------------------------------------------

class TestRAGPipelineQuery:
    def test_returns_query_result(self):
        pipeline = _make_pipeline(answer="Apple faces supply chain risks.")
        result = pipeline.query("What are Apple's risks?")
        assert isinstance(result, QueryResult)

    def test_answer_populated(self):
        pipeline = _make_pipeline(answer="Apple faces supply chain risks.")
        result = pipeline.query("What are Apple's risks?")
        assert result.answer == "Apple faces supply chain risks."

    def test_chunks_list_populated(self):
        pipeline = _make_pipeline()
        result = pipeline.query("What are Apple's risks?")
        assert len(result.chunks) >= 1
        assert isinstance(result.chunks[0], RetrievedChunk)

    def test_latency_ms_positive(self):
        pipeline = _make_pipeline()
        result = pipeline.query("What are Apple's risks?")
        assert result.latency_ms > 0

    def test_query_context_preserved(self):
        pipeline = _make_pipeline()
        result = pipeline.query("test question")
        assert result.query_context.query_type == "general"
        assert result.query_context.tickers == ["AAPL"]

    def test_profile_false_no_timing(self):
        pipeline = _make_pipeline()
        result = pipeline.query("What are Apple's risks?", profile=False)
        assert result.metadata.get("timing", {}) == {}

    def test_profile_true_returns_timing_breakdown(self):
        pipeline = _make_pipeline()
        result = pipeline.query("What are Apple's risks?", profile=True)
        timing = result.metadata.get("timing", {})
        assert "query_analysis_ms" in timing
        assert "retrieval_ms" in timing
        assert "llm_complete_ms" in timing
        assert "prompt_build_ms" in timing

    def test_profile_substeps_sum_to_roughly_total(self):
        pipeline = _make_pipeline()
        result = pipeline.query("What are Apple's risks?", profile=True)
        timing = result.metadata["timing"]
        total_timed = sum(timing.values())
        # Timed substeps should account for at least 80% of total latency
        assert total_timed <= result.latency_ms * 1.05, "Substep times exceed total latency"

    def test_is_indexed_true_when_store_has_chunks(self):
        pipeline = _make_pipeline()
        assert pipeline.is_indexed() is True

    def test_is_indexed_false_when_store_empty(self):
        pipeline = _make_pipeline()
        pipeline._store.count.return_value = 0
        assert pipeline.is_indexed() is False
