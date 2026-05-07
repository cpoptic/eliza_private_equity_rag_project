"""
Section-aware chunker for SEC 10-K and 10-Q filings.

Strategy:
  1. Detect Item boundaries using a two-pass approach:
       Pass A — reject TOC entries (lines containing | and a trailing page number)
       Pass B — accept real section headers (standalone Item lines in body text)
  2. Group text between consecutive Item headers into sections.
  3. If a section exceeds CHUNK_TOKEN_LIMIT, split it at paragraph boundaries
     with CHUNK_OVERLAP_TOKENS of context carry-over.
  4. Attach full provenance metadata to every chunk.

10-K items covered (high-value for RAG):
  Part I:  1 (Business), 1A (Risk Factors), 1B (Unresolved Staff Comments),
           2 (Properties), 3 (Legal Proceedings)
  Part II: 5 (Market/Equity), 7 (MD&A), 7A (Market Risk), 8 (Financial Stmts)

10-Q items covered:
  Part I:  1 (Financial Statements), 2 (MD&A), 3 (Market Risk), 4 (Controls)
  Part II: 1 (Legal), 1A (Risk Factors)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

from src.interfaces import BaseChunker, Chunk, FilingMetadata


class _FallbackEncoder:
    """
    Word-split token estimator used when tiktoken BPE vocab is unavailable
    (e.g. network-restricted environments, CI without model downloads).
    Approximates cl100k_base at ~1.3 tokens/word.
    """
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_TOKEN_LIMIT = 800
DEFAULT_OVERLAP_TOKENS = 100

# Sections we want to index (skip boilerplate-only items)
_HIGH_VALUE_SECTIONS = {
    "10-K": {"1", "1a", "1b", "2", "3", "5", "7", "7a", "8"},
    "10-Q": {"1", "1a", "2", "3", "4"},
}

# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

# Matches actual section headers in the filing body.
# Patterns observed in corpus:
#   "Item 1A.    Risk Factors"          (4+ spaces after dot)
#   "Item 1A. Risk Factors"             (1 space)
#   "Item 1A.Risk Factors"              (no space)
#   "ITEM 1A. RISK FACTORS"             (all caps)
#   "Item 1A"                           (just the item number)
# The regex captures the item number (e.g. "1A") and optional title.
_SECTION_HEADER_RE = re.compile(
    r"^(?:PART\s+(?:I{1,3}|IV|V)\s*\n)?"   # optional PART prefix line
    r"Item\s+(1A?B?|[2-9]|1[0-6]|7A?)\b"    # "Item 1A" / "Item 7" etc.
    r"(?:\.?\s{0,6}([^\n|]{0,80}))?$",       # optional title (no pipes — excludes TOC)
    re.IGNORECASE | re.MULTILINE,
)

# TOC entry detector: lines with pipe characters and trailing page numbers.
# Example: "Item 1A. | Risk Factors | 5"
_TOC_LINE_RE = re.compile(r"\|.*\|\s*\d+\s*$")

# Known item number → canonical display name
_ITEM_NAMES: dict[str, str] = {
    "1":   "Business",
    "1a":  "Risk Factors",
    "1b":  "Unresolved Staff Comments",
    "2":   "Properties",
    "3":   "Legal Proceedings",
    "4":   "Mine Safety Disclosures",
    "5":   "Market for Common Equity",
    "6":   "Selected Financial Data",
    "7":   "MD&A",
    "7a":  "Quantitative and Qualitative Disclosures About Market Risk",
    "8":   "Financial Statements",
    "9":   "Changes in Disagreements with Accountants",
    "9a":  "Controls and Procedures",
    "10":  "Directors and Executive Officers",
    "15":  "Exhibits",
}


@dataclass
class _Section:
    item_key: str       # normalised: "1a", "7", etc.
    display_name: str   # "Item 1A - Risk Factors"
    order: int
    text: str
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Chunker implementation
# ---------------------------------------------------------------------------

class SectionAwareChunker(BaseChunker):
    """
    Splits a filing into chunks aligned to SEC Item section boundaries.

    Args:
        chunk_token_limit: Maximum tokens per chunk before splitting.
        overlap_tokens: Tokens of context to carry over at split boundaries.
        filing_type_filter: If set, only process this filing type (e.g. "10-K").
        high_value_only: If True, skip items not in _HIGH_VALUE_SECTIONS.
    """

    def __init__(
        self,
        chunk_token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        high_value_only: bool = True,
    ) -> None:
        self.chunk_token_limit = chunk_token_limit
        self.overlap_tokens = overlap_tokens
        self.high_value_only = high_value_only
        if _TIKTOKEN_AVAILABLE:
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = _FallbackEncoder()
        else:
            self._enc = _FallbackEncoder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: FilingMetadata) -> list[Chunk]:
        sections = self._detect_sections(text, metadata.filing_type)
        if not sections:
            # Fallback: treat whole document as one section
            sections = [_Section(
                item_key="full",
                display_name="Full Document",
                order=0,
                text=text,
                char_start=0,
                char_end=len(text),
            )]

        chunks: list[Chunk] = []
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)

        return chunks

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    def _detect_sections(self, text: str, filing_type: str) -> list[_Section]:
        """
        Two-pass section boundary detection.

        Pass 1: Find all regex matches for Item headers.
        Pass 2: Discard matches that fall on TOC lines.
        """
        wanted = _HIGH_VALUE_SECTIONS.get(filing_type, set())
        lines = text.split("\n")
        line_offsets = _compute_line_offsets(text)

        # Pass 1: collect candidate matches
        candidates: list[tuple[int, str, str]] = []  # (char_pos, item_key, raw_title)
        for m in _SECTION_HEADER_RE.finditer(text):
            item_num = m.group(1).lower().replace(" ", "")
            raw_title = (m.group(2) or "").strip()

            if self.high_value_only and item_num not in wanted:
                continue

            # Pass 2: reject TOC entries
            line_idx = _char_to_line(m.start(), line_offsets)
            if line_idx is not None and _TOC_LINE_RE.search(lines[line_idx]):
                continue

            candidates.append((m.start(), item_num, raw_title))

        if not candidates:
            return []

        # Deduplicate: for each item_key keep the first occurrence
        seen: set[str] = set()
        unique: list[tuple[int, str, str]] = []
        for pos, key, title in candidates:
            if key not in seen:
                seen.add(key)
                unique.append((pos, key, title))

        # Sort by position and build _Section objects
        unique.sort(key=lambda x: x[0])
        sections: list[_Section] = []
        for i, (pos, key, raw_title) in enumerate(unique):
            end_pos = unique[i + 1][0] if i + 1 < len(unique) else len(text)
            title = self._resolve_title(key, raw_title)
            sections.append(_Section(
                item_key=key,
                display_name=f"Item {key.upper()} - {title}",
                order=i,
                text=text[pos:end_pos].strip(),
                char_start=pos,
                char_end=end_pos,
            ))

        return sections

    def _resolve_title(self, item_key: str, raw_title: str) -> str:
        """Return canonical title, falling back to raw text from the header match."""
        canonical = _ITEM_NAMES.get(item_key.lower())
        if canonical:
            return canonical
        # Clean up raw title: strip trailing punctuation / whitespace
        clean = re.sub(r"[.\s]+$", "", raw_title).strip()
        return clean if clean else f"Item {item_key.upper()}"

    # ------------------------------------------------------------------
    # Token-aware splitting
    # ------------------------------------------------------------------

    def _chunk_section(
        self, section: _Section, metadata: FilingMetadata
    ) -> list[Chunk]:
        """
        Split one section into token-bounded chunks at paragraph boundaries.
        """
        tokens = self._enc.encode(section.text)
        if len(tokens) <= self.chunk_token_limit:
            # Section fits in one chunk
            return [self._make_chunk(section.text, section, metadata, 0)]

        # Split into paragraphs and accumulate
        paragraphs = re.split(r"\n{2,}", section.text)
        chunks: list[Chunk] = []
        current_paras: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(self._enc.encode(para))

            if current_tokens + para_tokens > self.chunk_token_limit and current_paras:
                # Emit current accumulation
                chunk_text = "\n\n".join(current_paras)
                chunks.append(self._make_chunk(chunk_text, section, metadata, chunk_idx))
                chunk_idx += 1

                # Carry overlap: last N tokens worth of text
                overlap_text = self._tail_tokens(chunk_text, self.overlap_tokens)
                current_paras = [overlap_text, para] if overlap_text else [para]
                current_tokens = len(self._enc.encode("\n\n".join(current_paras)))
            else:
                current_paras.append(para)
                current_tokens += para_tokens

        # Emit remainder
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(self._make_chunk(chunk_text, section, metadata, chunk_idx))

        return chunks

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        """Return the last n_tokens worth of text (approximate, by word boundary)."""
        tokens = self._enc.encode(text)
        if len(tokens) <= n_tokens:
            return text
        tail_tokens = tokens[-n_tokens:]
        return self._enc.decode(tail_tokens)

    def _make_chunk(
        self,
        text: str,
        section: _Section,
        metadata: FilingMetadata,
        idx: int,
    ) -> Chunk:
        tokens = self._enc.encode(text)
        chunk_id = _make_chunk_id(metadata, section.item_key, idx)
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
            section=section.display_name,
            section_order=section.order * 100 + idx,
            token_count=len(tokens),
            char_start=section.char_start,
            char_end=section.char_end,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_line_offsets(text: str) -> list[int]:
    """Return the character offset of the start of each line."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def _char_to_line(char_pos: int, offsets: list[int]) -> int | None:
    """Binary search for the line index of a character offset."""
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= char_pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi if hi >= 0 else None


def _make_chunk_id(metadata: FilingMetadata, item_key: str, idx: int) -> str:
    """Deterministic chunk ID: hash of (ticker, period, item, idx)."""
    raw = f"{metadata.ticker}_{metadata.report_period}_{metadata.filing_type}_{item_key}_{idx}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{metadata.ticker}_{metadata.filing_type}_{item_key}_{idx}_{digest}"
