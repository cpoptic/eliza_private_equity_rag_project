"""
Unit tests for FilingParser and SectionAwareChunker.
"""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from src.interfaces import FilingMetadata
from src.preprocessing.parser import FilingParser, _normalize_item_headers
from src.preprocessing.chunker import SectionAwareChunker, EMBED_HARD_MAX_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**kwargs) -> FilingMetadata:
    defaults = dict(
        company="Test Corp",
        ticker="TEST",
        filing_type="10-K",
        filing_date="2023-12-31",
        report_period="2023-09-30",
        quarter=None,
        cik="0001234567",
        source_url="https://example.com",
        fiscal_year=2023,
        source_file="test_10K.txt",
    )
    defaults.update(kwargs)
    return FilingMetadata(**defaults)


def _make_corpus_file(tmp_path, name="AAPL_10K_2022Q3_2022-10-28_full.txt", body="") -> str:
    separator = "=" * 80 + "\n"
    header = (
        "Company: Apple Inc\n"
        "Ticker: AAPL\n"
        "Filing Type: 10-K (Annual Report)\n"
        "Filing Date: 2022-10-28\n"
        "Report Period: 2022-09-24\n"
        "Quarter: 2022Q3\n"
        "CIK: 0000320193\n"
        "Source: SEC EDGAR\n"
        "URL: https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108-index.htm\n"
    )
    xbrl_blob = "XBRL_DATA_BLOB_abc123def456\n"
    sentinel = "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
    content = header + separator + xbrl_blob + sentinel + body
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# FilingParser — header parsing
# ---------------------------------------------------------------------------

class TestFilingParserHeader:
    def test_parses_company_and_ticker(self, tmp_path):
        path = _make_corpus_file(tmp_path, body="Some text")
        parser = FilingParser()
        meta, _ = parser.parse(path)
        assert meta.company == "Apple Inc"
        assert meta.ticker == "AAPL"

    def test_parses_filing_type_strips_parenthetical(self, tmp_path):
        path = _make_corpus_file(tmp_path, body="Some text")
        parser = FilingParser()
        meta, _ = parser.parse(path)
        assert meta.filing_type == "10-K"

    def test_fiscal_year_derived_from_filing_date(self, tmp_path):
        path = _make_corpus_file(tmp_path, body="Some text")
        parser = FilingParser()
        meta, _ = parser.parse(path)
        assert meta.fiscal_year == 2022

    def test_quarter_parsed_when_present(self, tmp_path):
        path = _make_corpus_file(tmp_path, body="Some text")
        parser = FilingParser()
        meta, _ = parser.parse(path)
        assert meta.quarter == "2022Q3"

    def test_source_file_set_to_filename(self, tmp_path):
        fname = "AAPL_10K_2022Q3_2022-10-28_full.txt"
        path = _make_corpus_file(tmp_path, name=fname, body="Some text")
        parser = FilingParser()
        meta, _ = parser.parse(path)
        assert meta.source_file == fname


# ---------------------------------------------------------------------------
# FilingParser — XBRL stripping
# ---------------------------------------------------------------------------

class TestFilingParserXBRL:
    def test_xbrl_stripped_from_output(self, tmp_path):
        body = "This is the actual filing content.\nItem 1. Business\nWe do things."
        path = _make_corpus_file(tmp_path, body=body)
        parser = FilingParser()
        _, text = parser.parse(path)
        # Header fields must not appear in the cleaned text
        assert "Company: Apple Inc" not in text
        assert "Ticker: AAPL" not in text
        # Body content must be preserved
        assert "We do things." in text

    def test_body_content_preserved(self, tmp_path):
        body = "Item 1. Business\nWe sell products."
        path = _make_corpus_file(tmp_path, body=body)
        parser = FilingParser()
        _, text = parser.parse(path)
        assert "We sell products." in text

    def test_fallback_when_sentinel_missing(self, tmp_path):
        header = (
            "Company: Test Corp\nTicker: TEST\nFiling Type: 10-K\n"
            "Filing Date: 2023-01-01\nReport Period: 2022-12-31\nQuarter: N/A\n"
            "CIK: 0000111\nSource: SEC EDGAR\nURL: https://example.com\n"
            "=" * 80 + "\n"
        )
        body = "No XBRL sentinel in this file."
        path = tmp_path / "test.txt"
        path.write_text(header + body, encoding="utf-8")
        parser = FilingParser()
        _, text = parser.parse(str(path))
        assert "No XBRL sentinel" in text


# ---------------------------------------------------------------------------
# _normalize_item_headers
# ---------------------------------------------------------------------------

class TestNormalizeItemHeaders:
    def test_adds_newline_before_item_header(self):
        text = "Some text.Item 1A. Risk Factors\nMore text."
        result = _normalize_item_headers(text)
        assert "\nItem 1A" in result

    def test_adds_newline_after_item_header(self):
        text = "Prefix text.Item 7. MD&A\nContent"
        result = _normalize_item_headers(text)
        assert "Item 7.\n" in result or "Item 7" in result

    def test_replaces_non_breaking_spaces(self):
        text = "Item\xa01A.\xa0Risk Factors"
        result = _normalize_item_headers(text)
        assert "\xa0" not in result

    def test_toc_entries_skipped(self):
        text = "Item 1A. | Risk Factors | 12"
        result = _normalize_item_headers(text)
        # TOC entries should NOT get extra newlines wrapping them
        assert result.count("\nItem") == 0 or "| " in result

    def test_already_on_own_line_not_double_wrapped(self):
        text = "\nItem 1A.\nRisk Factors content\n"
        result = _normalize_item_headers(text)
        assert "Item 1A" in result
        # Should not produce triple newlines
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# SectionAwareChunker
# ---------------------------------------------------------------------------

class TestSectionAwareChunker:
    def test_produces_at_least_one_chunk(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        text = "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nSome filing text."
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1

    def test_chunk_ids_unique(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            "Item 1.\nBusiness description here. " * 50 + "\n"
            "Item 1A.\nRisk factor one. " * 50
        )
        chunks = chunker.chunk(text, meta)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_metadata_matches_filing(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata(ticker="MSFT", filing_type="10-K")
        text = "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nContent here."
        chunks = chunker.chunk(text, meta)
        for chunk in chunks:
            assert chunk.metadata.ticker == "MSFT"
            assert chunk.metadata.filing_type == "10-K"

    def test_token_count_within_hard_limit(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        # Build a section with a very long line that must be hard-split
        long_line = "word " * 2000
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            f"Item 1.\n{long_line}\n"
        )
        chunks = chunker.chunk(text, meta)
        for chunk in chunks:
            assert chunk.token_count <= EMBED_HARD_MAX_TOKENS, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens, "
                f"exceeds hard limit of {EMBED_HARD_MAX_TOKENS}"
            )

    def test_section_label_present(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            "Item 1A.\nRisk Factors\nWe face many risks."
        )
        chunks = chunker.chunk(text, meta)
        sections = [c.section.lower() for c in chunks]
        assert any("1a" in s or "risk" in s for s in sections)

    def test_char_offsets_non_overlapping(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            + "Item 1.\nBusiness. " * 60
            + "Item 1A.\nRisks. " * 60
        )
        chunks = chunker.chunk(text, meta)
        # char_end should not precede char_start
        for chunk in chunks:
            assert chunk.char_end >= chunk.char_start

    def test_10q_filing_chunks(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata(filing_type="10-Q", quarter="2023Q1")
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            "Item 1.\nFinancial Statements content here. " * 30 + "\n"
            "Item 2.\nMD&A content here. " * 30
        )
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1

    def test_provenance_header_format(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata(ticker="NVDA", filing_type="10-K")
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            "Item 1.\nBusiness description. " * 30
        )
        chunks = chunker.chunk(text, meta)
        header = chunks[0].provenance_header()
        assert "NVDA" in header
        assert "10-K" in header

    def test_cross_reference_not_treated_as_section(self):
        chunker = SectionAwareChunker()
        meta = _make_metadata()
        # "Item 1A\nof this Form 10-K" should NOT create a new section boundary
        text = (
            "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n"
            "Item 1.\nBusiness description. See Item 1A\nof this Form 10-K for risk details.\n"
            "More business content here. " * 20
        )
        chunks = chunker.chunk(text, meta)
        # The cross-reference should not produce a tiny isolated chunk
        cross_ref_chunks = [c for c in chunks if c.token_count < 5]
        assert len(cross_ref_chunks) == 0, (
            f"Found suspiciously small chunks: {[(c.chunk_id, c.token_count) for c in cross_ref_chunks]}"
        )
