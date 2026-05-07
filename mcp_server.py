"""
Optional MCP server — exposes the SEC filing RAG pipeline as tools
callable from Claude Desktop or any MCP client.

This is a BONUS layer on top of the existing pipeline.
The core Streamlit demo does not depend on this file.

Usage (after building the index):
    uv run python mcp_server.py

Then add to Claude Desktop's config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "sec-filings": {
          "command": "uv",
          "args": ["run", "python", "mcp_server.py"],
          "cwd": "/path/to/eliza-sec-rag"
        }
      }
    }

Then in Claude Desktop you can ask:
    "Search the SEC filings for NVIDIA's revenue guidance"
    "What risk factors does Apple disclose about supply chains?"
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from src.pipeline import build_pipeline

mcp = FastMCP("SEC Filing Intelligence")
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@mcp.tool()
def query_filings(question: str) -> str:
    """
    Ask a natural-language question about SEC filings (10-K and 10-Q).

    Searches across Apple, Amazon, Google, Microsoft, NVIDIA, Tesla, JPMorgan,
    Bank of America, Disney, Meta, Pfizer, Coca-Cola, ExxonMobil, and UnitedHealth.

    Examples:
    - "What are Apple's primary risk factors?"
    - "How has NVIDIA's revenue trended over the last two years?"
    - "Compare Microsoft and Google's cloud strategies"
    - "What cybersecurity risks do major banks disclose?"

    Args:
        question: A natural-language business question about the companies.

    Returns:
        A structured analysis grounded in SEC filing excerpts with citations.
    """
    pipeline = _get_pipeline()

    if not pipeline._store.collection_exists():
        return (
            "The filing index has not been built yet. "
            "Run: uv run python scripts/build_index.py"
        )

    result = pipeline.query(question)
    return result.answer


@mcp.tool()
def list_indexed_companies() -> str:
    """
    List all companies currently indexed and available for querying.

    Returns company names, tickers, filing types, and year coverage.
    """
    pipeline = _get_pipeline()

    if not pipeline._store.collection_exists():
        return "No index found. Run: uv run python scripts/build_index.py"

    chunks = pipeline._store.get_all_chunks()
    if not chunks:
        return "Index is empty."

    from collections import defaultdict
    by_ticker: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_ticker[c.metadata.ticker].append(c)

    lines = ["Companies indexed in the SEC filing knowledge base:\n"]
    for ticker in sorted(by_ticker.keys()):
        tc = by_ticker[ticker]
        company = tc[0].metadata.company
        types = sorted({c.metadata.filing_type for c in tc})
        years = sorted({c.metadata.fiscal_year for c in tc})
        year_range = f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
        lines.append(
            f"  {ticker:6s} {company[:35]:35s} "
            f"{', '.join(types):10s} {year_range}"
        )

    lines.append(f"\nTotal chunks indexed: {len(chunks):,}")
    return "\n".join(lines)


@mcp.tool()
def get_filing_sections(ticker: str, filing_type: str = "10-K") -> str:
    """
    List the sections available for a specific company's most recent filing.

    Useful for understanding what topics are indexed before asking a question.

    Args:
        ticker: Company ticker symbol (e.g. "AAPL", "NVDA", "MSFT")
        filing_type: "10-K" for annual reports or "10-Q" for quarterly reports
    """
    pipeline = _get_pipeline()
    chunks = pipeline._store.get_all_chunks()

    ticker = ticker.upper()
    relevant = [
        c for c in chunks
        if c.metadata.ticker == ticker and c.metadata.filing_type == filing_type
    ]

    if not relevant:
        available = sorted({c.metadata.ticker for c in chunks})
        return (
            f"No {filing_type} chunks found for {ticker}. "
            f"Available tickers: {', '.join(available)}"
        )

    # Most recent period
    latest_period = max(c.metadata.report_period for c in relevant)
    latest_chunks = [c for c in relevant if c.metadata.report_period == latest_period]

    sections = sorted({c.section for c in latest_chunks}, key=lambda s: latest_chunks[0].section_order)
    company = latest_chunks[0].metadata.company

    lines = [
        f"{company} ({ticker}) — {filing_type} — Period: {latest_period[:7]}\n",
        "Indexed sections:",
    ]
    for section in sections:
        count = sum(1 for c in latest_chunks if c.section == section)
        lines.append(f"  • {section}  ({count} chunk{'s' if count != 1 else ''})")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
