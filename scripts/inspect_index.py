#!/usr/bin/env python3
"""
Inspect the built index: companies, filing types, date coverage, chunk counts.

Run before the demo to verify the index is complete and correct.

Usage:
    uv run python scripts/inspect_index.py
    uv run python scripts/inspect_index.py --ticker AAPL
    uv run python scripts/inspect_index.py --sample-chunks NVDA "Item 1A"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the SEC filing index")
    parser.add_argument("--ticker", help="Show detailed breakdown for a specific ticker")
    parser.add_argument("--sample-chunks", nargs=2, metavar=("TICKER", "SECTION"),
                        help="Print sample chunk text: --sample-chunks AAPL 'Item 1A'")
    args = parser.parse_args()

    from src.pipeline import build_pipeline
    pipeline = build_pipeline()
    store = pipeline._store

    if not store.collection_exists():
        console.print("[red]✗ No index found. Run: uv run python scripts/build_index.py[/red]")
        sys.exit(1)

    total = store.count()
    console.print(Panel(
        f"[bold yellow]SEC Filing Index Inspection[/bold yellow]\n"
        f"[dim]Total chunks: {total:,}[/dim]",
        border_style="yellow",
    ))

    # Load all chunks for analysis
    with console.status("[dim]Loading chunks from store…[/dim]"):
        chunks = store.get_all_chunks()

    if args.sample_chunks:
        _show_sample_chunks(chunks, args.sample_chunks[0], args.sample_chunks[1])
        return

    if args.ticker:
        _show_ticker_detail(chunks, args.ticker.upper())
        return

    # ── Company / filing coverage ─────────────────────────────────────
    by_ticker: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_ticker[c.metadata.ticker].append(c)

    coverage_table = Table(
        "Ticker", "Company", "Filing Types", "Years Covered", "Chunks",
        box=box.SIMPLE_HEAVY,
        header_style="bold yellow",
        show_footer=True,
    )

    for ticker in sorted(by_ticker.keys()):
        tc = by_ticker[ticker]
        company = tc[0].metadata.company
        types = ", ".join(sorted({c.metadata.filing_type for c in tc}))
        years = sorted({c.metadata.fiscal_year for c in tc})
        year_str = f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
        coverage_table.add_row(ticker, company[:30], types, year_str, str(len(tc)))

    coverage_table.columns[4].footer = f"[bold]{total:,}[/bold]"
    console.print(coverage_table)

    # ── Section distribution ──────────────────────────────────────────
    section_counts = Counter(c.section for c in chunks)
    section_table = Table("Section", "Chunk Count", box=box.SIMPLE, header_style="bold")
    for section, count in section_counts.most_common(15):
        bar = "█" * (count // max(1, total // 80))
        section_table.add_row(section[:55], f"{count:,}  {bar}")
    console.print()
    console.print("[bold]Top Sections[/bold]")
    console.print(section_table)

    # ── Token distribution ────────────────────────────────────────────
    token_counts = [c.token_count for c in chunks]
    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        console.print(
            f"\n[dim]Token stats:[/dim] avg={avg:.0f}  "
            f"min={min(token_counts)}  max={max(token_counts)}  "
            f"total={sum(token_counts):,}"
        )


def _show_ticker_detail(chunks: list, ticker: str) -> None:
    tc = [c for c in chunks if c.metadata.ticker == ticker]
    if not tc:
        console.print(f"[red]No chunks found for ticker {ticker}[/red]")
        console.print(f"Available: {sorted({c.metadata.ticker for c in chunks})}")
        return

    console.print(f"\n[bold]{ticker}[/bold] — {tc[0].metadata.company}")
    console.print(f"Total chunks: {len(tc)}\n")

    by_period: dict[str, list] = defaultdict(list)
    for c in tc:
        by_period[c.metadata.report_period[:7]].append(c)

    table = Table("Period", "Filing Type", "Sections", "Chunks", box=box.SIMPLE)
    for period in sorted(by_period.keys()):
        pc = by_period[period]
        ft = pc[0].metadata.filing_type
        sections = len({c.section for c in pc})
        table.add_row(period, ft, str(sections), str(len(pc)))
    console.print(table)


def _show_sample_chunks(chunks: list, ticker: str, section_hint: str) -> None:
    matches = [
        c for c in chunks
        if c.metadata.ticker.upper() == ticker.upper()
        and section_hint.lower() in c.section.lower()
    ]
    if not matches:
        console.print(f"[red]No chunks found for {ticker} / '{section_hint}'[/red]")
        return

    console.print(f"\nShowing {min(3, len(matches))} sample chunks for "
                  f"[bold]{ticker}[/bold] / [bold]{section_hint}[/bold]\n")
    for c in matches[:3]:
        console.print(Panel(
            c.text[:600] + ("…" if len(c.text) > 600 else ""),
            title=f"[yellow]{c.provenance_header()}[/yellow]  tokens={c.token_count}",
            border_style="dim",
        ))


if __name__ == "__main__":
    main()
