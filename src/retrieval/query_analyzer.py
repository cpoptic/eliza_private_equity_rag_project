"""
Rule-based query analyzer.

Extracts tickers, temporal range, section hints, and query type from a
natural-language question — no LLM calls (reserved for final answer only).
"""

from __future__ import annotations

import re
from datetime import datetime

from src.interfaces import BaseQueryAnalyzer, QueryContext

# ---------------------------------------------------------------------------
# Ticker knowledge base
# ---------------------------------------------------------------------------

# All tickers present in the corpus
_CORPUS_TICKERS: frozenset[str] = frozenset({
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS", "BRK", "BRK.B",
    "JNJ", "PFE", "UNH", "TMO", "ABBV", "MRK",
    "XOM", "CVX",
    "HD", "WMT", "COST", "TGT", "AMZN",
    "DIS", "NFLX", "KO", "PEP",
    "ORCL", "CRM", "ADBE", "INTC", "AMD",
})

# Company name / alias → ticker symbol
_TICKER_ALIASES: dict[str, str] = {
    # Tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOG",
    "alphabet": "GOOG",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "netflix": "NFLX",
    # Finance
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "chase": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "goldman": "GS",
    "morgan stanley": "MS",
    "berkshire": "BRK",
    "berkshire hathaway": "BRK",
    # Healthcare
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "thermo fisher": "TMO",
    "thermofisher": "TMO",
    "abbvie": "ABBV",
    "merck": "MRK",
    # Energy
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    # Consumer
    "home depot": "HD",
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "disney": "DIS",
    "coca-cola": "KO",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
}
# Add each corpus ticker as its own alias (case-insensitive lookup)
for _t in _CORPUS_TICKERS:
    _TICKER_ALIASES[_t.lower()] = _t

# ---------------------------------------------------------------------------
# Section keyword → Item mapping
# ---------------------------------------------------------------------------

_KEYWORD_TO_SECTION: dict[str, str] = {
    "risk": "Item 1A",
    "risks": "Item 1A",
    "risk factor": "Item 1A",
    "risk factors": "Item 1A",
    "revenue": "Item 7",
    "revenues": "Item 7",
    "sales": "Item 7",
    "earnings": "Item 7",
    "income": "Item 7",
    "profit": "Item 7",
    "margin": "Item 7",
    "mda": "Item 7",
    "md&a": "Item 7",
    "management discussion": "Item 7",
    "financial statements": "Item 8",
    "financial statement": "Item 8",
    "balance sheet": "Item 8",
    "cash flow": "Item 8",
    "business": "Item 1",
    "operations": "Item 1",
    "products": "Item 1",
    "services": "Item 1",
    "legal": "Item 3",
    "litigation": "Item 3",
    "lawsuit": "Item 3",
    "regulatory": "Item 1A",
    "market risk": "Item 7A",
    "interest rate": "Item 7A",
    "foreign exchange": "Item 7A",
}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_LAST_N_YEARS_RE = re.compile(r"last\s+(\w+|\d+)\s+years?", re.IGNORECASE)
_SINCE_YEAR_RE = re.compile(r"since\s+(20\d{2})", re.IGNORECASE)
_IN_YEAR_RE = re.compile(r"\bin\s+(20\d{2})\b", re.IGNORECASE)

_WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

_COMPARISON_KEYWORDS = {"compare", "comparison", "vs", "versus", "compared to",
                        "differences between", "contrast", "relative to"}
_TREND_KEYWORDS = {"trend", "trends", "over time", "changed", "changes", "growth",
                   "decline", "declined", "grew", "trajectory", "historical",
                   "year over year", "yoy", "quarter over quarter", "qoq"}
_THEMATIC_KEYWORDS = {"pharma", "pharmaceutical", "sector", "industry", "companies",
                      "healthcare sector", "tech sector", "banks", "financials"}


class RuleBasedQueryAnalyzer(BaseQueryAnalyzer):

    def analyze(self, question: str) -> QueryContext:
        q_lower = question.lower()

        tickers = self._extract_tickers(question, q_lower)
        year_range = self._extract_year_range(question, q_lower)
        section_hints = self._extract_section_hints(q_lower)
        query_type = self._classify(q_lower, tickers, year_range)
        sub_queries = self._build_sub_queries(question, tickers, query_type)

        return QueryContext(
            original_query=question,
            tickers=tickers,
            year_range=year_range,
            section_hints=section_hints,
            query_type=query_type,
            sub_queries=sub_queries,
        )

    # ------------------------------------------------------------------
    # Ticker extraction
    # ------------------------------------------------------------------

    def _extract_tickers(self, question: str, q_lower: str) -> list[str]:
        found: set[str] = set()

        # Match explicit ticker symbols (1-5 uppercase letters)
        for token in re.findall(r"\b([A-Z]{1,5})\b", question):
            canonical = _TICKER_ALIASES.get(token.lower())
            if canonical and canonical in _CORPUS_TICKERS:
                found.add(canonical)

        # Multi-word alias lookup with word boundaries (longest match first)
        aliases_by_len = sorted(_TICKER_ALIASES.keys(), key=len, reverse=True)
        for alias in aliases_by_len:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, q_lower):
                ticker = _TICKER_ALIASES[alias]
                if ticker in _CORPUS_TICKERS:
                    found.add(ticker)

        return sorted(found)

    # ------------------------------------------------------------------
    # Temporal extraction
    # ------------------------------------------------------------------

    def _extract_year_range(self, question: str, q_lower: str) -> tuple[int, int] | None:
        current_year = datetime.now().year

        # "last N years"
        m = _LAST_N_YEARS_RE.search(q_lower)
        if m:
            raw = m.group(1).lower()
            n = _WORD_TO_NUM.get(raw) or (int(raw) if raw.isdigit() else 2)
            return (current_year - n, current_year)

        # "since YYYY"
        m = _SINCE_YEAR_RE.search(question)
        if m:
            return (int(m.group(1)), current_year)

        # "in YYYY"
        m = _IN_YEAR_RE.search(question)
        if m:
            y = int(m.group(1))
            return (y, y)

        # General year mentions
        years = [int(y) for y in _YEAR_RE.findall(question)]
        if years:
            return (min(years), max(years))

        return None

    # ------------------------------------------------------------------
    # Section hints
    # ------------------------------------------------------------------

    def _extract_section_hints(self, q_lower: str) -> list[str]:
        hints: set[str] = set()
        for keyword, section in _KEYWORD_TO_SECTION.items():
            if keyword in q_lower:
                hints.add(section)
        return sorted(hints)

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        q_lower: str,
        tickers: list[str],
        year_range: tuple[int, int] | None,
    ) -> str:
        has_comparison = any(kw in q_lower for kw in _COMPARISON_KEYWORDS)
        if len(tickers) >= 2 or has_comparison:
            return "comparison"

        has_trend = any(kw in q_lower for kw in _TREND_KEYWORDS)
        has_multi_year = year_range is not None and year_range[1] - year_range[0] >= 1
        if has_trend or has_multi_year:
            return "trend"

        has_thematic = any(kw in q_lower for kw in _THEMATIC_KEYWORDS)
        if has_thematic and not tickers:
            return "thematic"

        return "general"

    # ------------------------------------------------------------------
    # Sub-query generation
    # ------------------------------------------------------------------

    def _build_sub_queries(
        self,
        question: str,
        tickers: list[str],
        query_type: str,
    ) -> list[str]:
        if query_type != "comparison" or not tickers:
            return [question]

        # Strip ticker tokens from the question to get the focus
        focus = question
        for ticker in tickers:
            focus = re.sub(rf"\b{re.escape(ticker)}\b", "", focus, flags=re.IGNORECASE)
        # Also strip known company name aliases (skip short ones like "amd" to avoid over-stripping)
        for alias, t in _TICKER_ALIASES.items():
            if t in tickers and len(alias) > 4:
                focus = re.sub(rf"\b{re.escape(alias)}\b", "", focus, flags=re.IGNORECASE)
        # Remove dangling connectors left after stripping names
        focus = re.sub(r"\b(and|or|,)\b", " ", focus, flags=re.IGNORECASE)
        focus = re.sub(r"[,;]+", " ", focus)
        focus = re.sub(r"\s{2,}", " ", focus).strip(" ,;-")

        return [f"{ticker}: {focus}" for ticker in tickers]
