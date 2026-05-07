"""
Filing parser: extracts structured metadata and clean text from SEC corpus files.

Each file in edgar_corpus/ has a 10-line header block followed by raw EDGAR
content that begins with an XBRL data blob. The parser:
  1. Reads the 10-line header → FilingMetadata
  2. Strips the XBRL blob (everything after the === separator until the
     "UNITED STATES / SECURITIES AND EXCHANGE COMMISSION" sentinel)
  3. Returns (FilingMetadata, cleaned_text) ready for the chunker
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.interfaces import FilingMetadata

logger = logging.getLogger(__name__)

# Sentinel that marks the end of the XBRL blob and start of readable filing text.
# In practice XBRL data is packed with no whitespace, so the sentinel appears as
# "UNITED STATESSECURITIES AND EXCHANGE COMMISSION" (zero or more spaces/newlines).
# Matches Item headers that aren't already at the start of a line.
# Handles non-breaking spaces (\xa0) used as padding in some filings.
_ITEM_HEADER_RE = re.compile(
    r"(?<!\n)(Item\s+(?:1[A-Za-z]?|[2-9]|1[0-6]|7[Aa]?)[\.\s])",
    re.IGNORECASE,
)


def _normalize_item_headers(text: str) -> str:
    """Ensure every Item header starts on its own line for the chunker regex."""
    return _ITEM_HEADER_RE.sub(r"\n\1", text)


_XBRL_END_RE = re.compile(
    r"UNITED\s+STATES\s*SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
    re.IGNORECASE,
)


class FilingParser:
    """Parses a single SEC corpus file into (FilingMetadata, cleaned_text)."""

    def parse(self, filepath: str | Path) -> tuple[FilingMetadata, str]:
        path = Path(filepath)
        raw = path.read_text(encoding="utf-8", errors="replace")

        metadata = self._parse_header(raw, source_file=path.name)
        cleaned = self._strip_xbrl(raw, source_file=path.name)

        return metadata, cleaned

    # ------------------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------------------

    def _parse_header(self, raw: str, source_file: str) -> FilingMetadata:
        lines = raw.splitlines()

        def _field(line: str) -> str:
            return line.split(":", 1)[1].strip() if ":" in line else line.strip()

        company       = _field(lines[0])
        ticker        = _field(lines[1])
        filing_type   = _field(lines[2]).split()[0]   # "10-K" from "10-K (Annual Report)"
        filing_date   = _field(lines[3])
        report_period = _field(lines[4])
        quarter_raw   = _field(lines[5])
        cik           = _field(lines[6])
        # lines[7] = "Source: SEC EDGAR"
        source_url    = _field(lines[8])

        quarter: str | None = quarter_raw if quarter_raw and quarter_raw.upper() != "N/A" else None
        fiscal_year = int(filing_date[:4])

        return FilingMetadata(
            company=company,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            report_period=report_period,
            quarter=quarter,
            cik=cik,
            source_url=source_url,
            fiscal_year=fiscal_year,
            source_file=source_file,
        )

    # ------------------------------------------------------------------
    # XBRL stripping
    # ------------------------------------------------------------------

    def _strip_xbrl(self, raw: str, source_file: str) -> str:
        sep_idx = raw.find("====")
        if sep_idx == -1:
            logger.warning("%s: header separator not found; using full text", source_file)
            return raw

        newline_after_sep = raw.find("\n", sep_idx)
        after_header = raw[newline_after_sep + 1:]

        match = _XBRL_END_RE.search(after_header)
        if match is None:
            logger.warning(
                "%s: XBRL sentinel not found; returning text after header separator",
                source_file,
            )
            return after_header.strip()

        cleaned = after_header[match.start():].strip()
        return _normalize_item_headers(cleaned)
