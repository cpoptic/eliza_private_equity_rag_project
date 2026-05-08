"""
Chunk coverage validator — analyses how well the chunker covers each SEC filing.

Outputs a Rich table to the terminal and saves a Markdown report under
reports/validate_chunks_reports/ with a timestamp.

Usage:
    uv run python scripts/validate_chunks.py [--corpus-dir edgar_corpus/]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from src.preprocessing.parser import FilingParser
from src.preprocessing.chunker import SectionAwareChunker, _HIGH_VALUE_SECTIONS

console = Console()

_SECTION_NAMES = {
    "1":  "Business / Financial Statements",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2":  "Properties / MD&A",
    "3":  "Legal / Quantitative Market Risk",
    "4":  "Controls and Procedures",
    "5":  "Market for Common Equity",
    "7":  "MD&A",
    "7a": "Quantitative Market Risk",
    "8":  "Financial Statements",
}


def _section_key(section_str: str) -> str:
    return section_str.split(" - ")[0].lower().replace("item ", "").strip()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_stats(corpus_dir: str) -> dict:
    parser = FilingParser()
    chunker = SectionAwareChunker()
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))

    # Per-type buckets
    data: dict[str, dict] = {
        "ALL": _empty_bucket(),
        "10-K": _empty_bucket(),
        "10-Q": _empty_bucket(),
    }
    errors: list[str] = []

    for fname in files:
        fpath = Path(corpus_dir) / fname
        try:
            metadata, text = parser.parse(fpath)
            chunks = chunker.chunk(text, metadata)
            ftype = metadata.filing_type
            buckets = [data["ALL"], data.setdefault(ftype, _empty_bucket())]

            n = len(chunks)
            found_keys = {_section_key(ch.section) for ch in chunks}
            wanted = _HIGH_VALUE_SECTIONS.get(ftype, set())
            token_counts = [ch.token_count for ch in chunks]

            for b in buckets:
                b["files"].append(fname)
                b["chunks_per_file"].append(n)
                b["token_counts"].extend(token_counts)
                b["ticker_chunks"][metadata.ticker].append(n)

                if "full" in found_keys:
                    b["full_doc_fallbacks"] += 1

                for key in wanted:
                    if key in found_keys:
                        b["section_hits"][key] += 1
                    else:
                        b["section_miss"][key] += 1

        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    data["errors"] = errors  # type: ignore[assignment]
    return data


def _empty_bucket() -> dict:
    return {
        "files": [],
        "chunks_per_file": [],
        "token_counts": [],
        "ticker_chunks": defaultdict(list),
        "section_hits": defaultdict(int),
        "section_miss": defaultdict(int),
        "full_doc_fallbacks": 0,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _stats_table(label: str, b: dict) -> Table:
    cpf = b["chunks_per_file"]
    tok = b["token_counts"]
    t = Table(title=f"{label} — Chunk Distribution", box=box.SIMPLE,
              show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Files", str(len(cpf)))
    t.add_row("Total chunks", f"{sum(cpf):,}")
    if cpf:
        t.add_row("Avg chunks / file", f"{statistics.mean(cpf):.1f}")
        t.add_row("Median chunks / file", f"{statistics.median(cpf):.1f}")
        t.add_row("Min / Max", f"{min(cpf)} / {max(cpf)}")
        t.add_row("Std dev", f"{statistics.stdev(cpf):.1f}" if len(cpf) > 1 else "—")
    if tok:
        t.add_row("Avg tokens / chunk", f"{statistics.mean(tok):.0f}")
        t.add_row("Max tokens / chunk", f"{max(tok):,}")
    t.add_row("Full-doc fallbacks", str(b["full_doc_fallbacks"]))
    return t


def _coverage_table(label: str, b: dict) -> Table:
    t = Table(title=f"{label} — Section Coverage", box=box.SIMPLE,
              show_header=True, header_style="bold magenta")
    t.add_column("Item", style="cyan")
    t.add_column("Section Name")
    t.add_column("Hit", justify="right", style="green")
    t.add_column("Miss", justify="right", style="red")
    t.add_column("Total", justify="right")
    t.add_column("Coverage", justify="right")

    all_keys = sorted(
        set(b["section_hits"]) | set(b["section_miss"]),
        key=lambda k: (len(k), k),
    )
    for key in all_keys:
        hits = b["section_hits"][key]
        miss = b["section_miss"][key]
        total = hits + miss
        pct = 100 * hits / total if total else 0
        color = "green" if pct >= 90 else ("yellow" if pct >= 70 else "red")
        t.add_row(
            f"Item {key.upper()}",
            _SECTION_NAMES.get(key, ""),
            str(hits), str(miss), str(total),
            f"[{color}]{pct:.0f}%[/{color}]",
        )
    return t


def _ticker_table(label: str, b: dict, top_n: int = 10) -> Table:
    t = Table(title=f"{label} — Per-Ticker (top {top_n})", box=box.SIMPLE,
              show_header=True, header_style="bold magenta")
    t.add_column("Ticker", style="cyan")
    t.add_column("Filings", justify="right")
    t.add_column("Total Chunks", justify="right")
    t.add_column("Avg / Filing", justify="right")

    sorted_tickers = sorted(
        b["ticker_chunks"].items(), key=lambda x: sum(x[1]), reverse=True
    )[:top_n]
    for ticker, counts in sorted_tickers:
        t.add_row(
            ticker,
            str(len(counts)),
            str(sum(counts)),
            f"{statistics.mean(counts):.1f}",
        )
    return t


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _md_stats(label: str, b: dict) -> list[str]:
    cpf = b["chunks_per_file"]
    tok = b["token_counts"]
    lines = [f"### {label} — Chunk Distribution", "", "| Metric | Value |", "|---|---|"]
    lines.append(f"| Files | {len(cpf)} |")
    lines.append(f"| Total chunks | {sum(cpf):,} |")
    if cpf:
        lines.append(f"| Avg chunks / file | {statistics.mean(cpf):.1f} |")
        lines.append(f"| Median chunks / file | {statistics.median(cpf):.1f} |")
        lines.append(f"| Min / Max chunks | {min(cpf)} / {max(cpf)} |")
    if tok:
        lines.append(f"| Avg tokens / chunk | {statistics.mean(tok):.0f} |")
        lines.append(f"| Max tokens / chunk | {max(tok):,} |")
    lines.append(f"| Full-doc fallbacks | {b['full_doc_fallbacks']} |")
    lines.append("")
    return lines


def _md_coverage(label: str, b: dict) -> list[str]:
    lines = [f"### {label} — Section Coverage", "",
             "| Item | Section | Hit | Miss | Total | Coverage |",
             "|---|---|---|---|---|---|"]
    all_keys = sorted(
        set(b["section_hits"]) | set(b["section_miss"]),
        key=lambda k: (len(k), k),
    )
    for key in all_keys:
        hits = b["section_hits"][key]
        miss = b["section_miss"][key]
        total = hits + miss
        pct = 100 * hits / total if total else 0
        lines.append(
            f"| Item {key.upper()} | {_SECTION_NAMES.get(key, '')} "
            f"| {hits} | {miss} | {total} | {pct:.0f}% |"
        )
    lines.append("")
    return lines


def _md_ticker(label: str, b: dict, top_n: int = 10) -> list[str]:
    lines = [f"### {label} — Per-Ticker Chunk Counts (top {top_n})", "",
             "| Ticker | Filings | Total Chunks | Avg / Filing |",
             "|---|---|---|---|"]
    for ticker, counts in sorted(
        b["ticker_chunks"].items(), key=lambda x: sum(x[1]), reverse=True
    )[:top_n]:
        lines.append(
            f"| {ticker} | {len(counts)} | {sum(counts):,} "
            f"| {statistics.mean(counts):.1f} |"
        )
    lines.append("")
    return lines


def write_report(data: dict, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_human = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path = reports_dir / f"validate_chunks_{ts_file}.md"

    md: list[str] = [f"# Chunk Validation Report — {ts_human}", ""]

    for label in ("ALL", "10-K", "10-Q"):
        if label not in data or not data[label]["files"]:
            continue
        md += [f"## {label}", ""]
        md += _md_stats(label, data[label])
        md += _md_coverage(label, data[label])
        md += _md_ticker(label, data[label])

    errors = data.get("errors", [])
    if errors:
        md += ["## Errors", ""]
        for e in errors:
            md.append(f"- {e}")
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_corpus(corpus_dir: str, reports_dir: Path) -> None:
    console.print(f"\n[bold]Validating corpus in {corpus_dir}...[/bold]")
    data = collect_stats(corpus_dir)

    for label in ("ALL", "10-K", "10-Q"):
        b = data.get(label, {})
        if not b or not b.get("files"):
            continue
        console.rule(f"[bold cyan]{label}[/bold cyan]")
        console.print(_stats_table(label, b))
        console.print(_coverage_table(label, b))
        console.print(_ticker_table(label, b))

    errors = data.get("errors", [])
    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")

    report_path = write_report(data, reports_dir)
    console.print(f"\n[dim]Report saved → {report_path}[/dim]\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate chunk coverage over SEC corpus")
    ap.add_argument(
        "--corpus-dir", default="edgar_corpus",
        help="Directory containing .txt corpus files (default: edgar_corpus)",
    )
    ap.add_argument(
        "--reports-dir", default="reports/validate_chunks_reports",
        help="Directory for Markdown reports (default: reports/validate_chunks_reports/)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir):
        console.print(f"[red]Corpus directory not found: {args.corpus_dir}[/red]")
        raise SystemExit(1)

    validate_corpus(args.corpus_dir, Path(args.reports_dir))


if __name__ == "__main__":
    main()
