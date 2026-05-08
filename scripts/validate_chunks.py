"""
Chunk coverage validator — analyses how well the chunker covers each SEC filing.

For each corpus file: parses + chunks, then reports which expected sections were
detected and how many chunks were produced. Aggregates into a distribution summary.

Usage:
    uv run python scripts/validate_chunks.py [--corpus-dir edgar_corpus/]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from src.preprocessing.parser import FilingParser
from src.preprocessing.chunker import SectionAwareChunker, _HIGH_VALUE_SECTIONS

console = Console()

_10K_SECTIONS = ["1", "1a", "1b", "2", "3", "5", "7", "7a", "8"]
_10Q_SECTIONS = ["1", "1a", "2", "3", "4"]


def _section_key(section_str: str) -> str:
    return section_str.split(" - ")[0].lower().replace("item ", "").strip()


def validate_corpus(corpus_dir: str) -> None:
    parser = FilingParser()
    chunker = SectionAwareChunker()

    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))

    # Per-file results
    chunks_per_file: list[int] = []
    section_hits: dict[str, int] = defaultdict(int)
    section_miss: dict[str, int] = defaultdict(int)
    full_doc_fallbacks: int = 0
    errors: list[str] = []

    # Per-ticker aggregates
    ticker_chunks: dict[str, list[int]] = defaultdict(list)

    console.print(f"\n[bold]Validating {len(files)} corpus files...[/bold]")

    for fname in files:
        fpath = Path(corpus_dir) / fname
        try:
            metadata, text = parser.parse(fpath)
            chunks = chunker.chunk(text, metadata)
            n = len(chunks)
            chunks_per_file.append(n)
            ticker_chunks[metadata.ticker].append(n)

            found_keys = {_section_key(ch.section) for ch in chunks}
            wanted = _HIGH_VALUE_SECTIONS.get(metadata.filing_type, set())

            if "full" in found_keys:
                full_doc_fallbacks += 1

            for key in wanted:
                if key in found_keys:
                    section_hits[key] += 1
                else:
                    section_miss[key] += 1

        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    # ── Summary stats ────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Chunk Distribution Summary[/bold cyan]")

    stats = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", justify="right")

    total = sum(chunks_per_file)
    avg = statistics.mean(chunks_per_file) if chunks_per_file else 0
    med = statistics.median(chunks_per_file) if chunks_per_file else 0
    mn = min(chunks_per_file) if chunks_per_file else 0
    mx = max(chunks_per_file) if chunks_per_file else 0
    stdev = statistics.stdev(chunks_per_file) if len(chunks_per_file) > 1 else 0

    stats.add_row("Files processed", str(len(files)))
    stats.add_row("Total chunks", f"{total:,}")
    stats.add_row("Avg chunks / file", f"{avg:.1f}")
    stats.add_row("Median chunks / file", f"{med:.1f}")
    stats.add_row("Min chunks / file", str(mn))
    stats.add_row("Max chunks / file", str(mx))
    stats.add_row("Std dev", f"{stdev:.1f}")
    stats.add_row("Full-doc fallbacks", str(full_doc_fallbacks))
    stats.add_row("Parse errors", str(len(errors)))

    console.print(stats)

    # ── Section coverage ─────────────────────────────────────────────────────
    console.rule("[bold cyan]Section Coverage[/bold cyan]")

    cov_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    cov_table.add_column("Item", style="cyan")
    cov_table.add_column("Section Name")
    cov_table.add_column("Hit", justify="right", style="green")
    cov_table.add_column("Miss", justify="right", style="red")
    cov_table.add_column("Total", justify="right")
    cov_table.add_column("Coverage", justify="right")

    _NAMES = {
        "1": "Business / Financial Stmts",
        "1a": "Risk Factors",
        "1b": "Unresolved Staff Comments",
        "2": "Properties / MD&A",
        "3": "Legal / Market Risk",
        "4": "Controls and Procedures",
        "5": "Market for Common Equity",
        "7": "MD&A",
        "7a": "Market Risk Disclosures",
        "8": "Financial Statements",
    }

    all_keys = sorted(
        set(section_hits.keys()) | set(section_miss.keys()),
        key=lambda k: (len(k), k),
    )
    for key in all_keys:
        hits = section_hits[key]
        miss = section_miss[key]
        total_k = hits + miss
        pct = 100 * hits / total_k if total_k else 0
        color = "green" if pct >= 90 else ("yellow" if pct >= 70 else "red")
        cov_table.add_row(
            f"Item {key.upper()}",
            _NAMES.get(key, ""),
            str(hits),
            str(miss),
            str(total_k),
            f"[{color}]{pct:.0f}%[/{color}]",
        )

    console.print(cov_table)

    # ── Per-ticker summary (top 10 by chunk count) ────────────────────────────
    console.rule("[bold cyan]Per-Ticker Chunk Counts (top 10)[/bold cyan]")

    ticker_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    ticker_table.add_column("Ticker", style="cyan")
    ticker_table.add_column("Filings", justify="right")
    ticker_table.add_column("Total Chunks", justify="right")
    ticker_table.add_column("Avg / Filing", justify="right")

    sorted_tickers = sorted(
        ticker_chunks.items(), key=lambda x: sum(x[1]), reverse=True
    )[:10]

    for ticker, chunk_counts in sorted_tickers:
        ticker_table.add_row(
            ticker,
            str(len(chunk_counts)),
            str(sum(chunk_counts)),
            f"{statistics.mean(chunk_counts):.1f}",
        )

    console.print(ticker_table)

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")

    console.print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate chunk coverage over SEC corpus")
    ap.add_argument(
        "--corpus-dir",
        default="edgar_corpus",
        help="Directory containing .txt corpus files (default: edgar_corpus)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir):
        console.print(f"[red]Corpus directory not found: {args.corpus_dir}[/red]")
        raise SystemExit(1)

    validate_corpus(args.corpus_dir)


if __name__ == "__main__":
    main()
