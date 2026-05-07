"""
Prompt builder for the RAG pipeline.

Constructs the system prompt and query-type-specific user prompts that are
passed to the single LLM call. Citation format: [TICKER | FORM | PERIOD | SECTION].
"""

from __future__ import annotations

from src.interfaces import RetrievedChunk

_SYSTEM_PROMPT = """\
You are a financial analyst assistant that answers questions exclusively from \
the SEC filing excerpts provided below.

Rules you must follow without exception:
1. Base every factual claim on the provided excerpts. Do not use prior knowledge \
to supply figures, dates, or conclusions that are not present in the excerpts.
2. After each claim, cite the source excerpt using the format \
[TICKER | FORM | PERIOD | SECTION] — e.g. [AAPL | 10-K | 2023-09 | Item 1A].
3. If the excerpts do not contain enough information to answer part of the question, \
say so explicitly: "The provided excerpts do not contain information about [topic]."
4. Do not speculate or extrapolate beyond what the excerpts state.
5. Be concise and precise. Prefer bullet points for lists; use prose for summaries."""

_QUERY_TYPE_INSTRUCTIONS: dict[str, str] = {
    "comparison": (
        "Structure your answer with a dedicated section for each company, then a "
        "brief comparative summary at the end. Use the company ticker as each section header."
    ),
    "trend": (
        "Present your findings in chronological order, explicitly citing the period "
        "(quarter or fiscal year) for each data point. Highlight directional changes."
    ),
    "thematic": (
        "Group your findings by theme rather than by company. For each theme, cite "
        "relevant examples from multiple companies where available."
    ),
    "general": (
        "Answer the question directly and concisely, citing the relevant excerpts."
    ),
}


class PromptBuilder:

    def build_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        query_type: str = "general",
    ) -> str:
        instruction = _QUERY_TYPE_INSTRUCTIONS.get(query_type, _QUERY_TYPE_INSTRUCTIONS["general"])

        excerpt_block = self._format_chunks(chunks)

        return (
            f"INSTRUCTION: {instruction}\n\n"
            f"EXCERPTS FROM SEC FILINGS:\n"
            f"{'—' * 60}\n"
            f"{excerpt_block}\n"
            f"{'—' * 60}\n\n"
            f"QUESTION: {question}"
        )

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for i, rc in enumerate(chunks, start=1):
            header = rc.chunk.provenance_header()
            parts.append(f"[Excerpt {i}] {header}\n{rc.chunk.text}")
        return "\n\n".join(parts)
