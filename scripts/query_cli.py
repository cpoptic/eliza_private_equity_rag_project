#!/usr/bin/env python3
"""
Terminal query interface for the SEC RAG pipeline.

Faster iteration loop than launching Streamlit for every test query.
Also useful for evaluating retrieval quality by inspecting raw chunks.

Usage:
    uv run python scripts/query_cli.py "What are Apple's risk factors?"
    uv run python scripts/query_cli.py --show-chunks "How has NVIDIA's revenue changed?"
    uv run python scripts/query_cli.py --chunks-only "Compare Apple and Tesla risks"
    uv run python scripts/query_cli.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def run_query(
    question: str, show_chunks: bool = False, chunks_only: bool = False
) -> None:
    from src.pipeline import build_pipeline

    pipeline = build_pipeline()

    if pipeline._store.count() == 0:
        console.print(
            "[red]✗ Index is empty. Run: uv run python scripts/build_index.py[/red]"
        )
        sys.exit(1)

    with console.status(f"[dim]Analyzing query…[/dim]"):
        result = pipeline.query(question)

    # ── Query metadata ────────────────────────────────────────────────
    meta_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta_table.add_column(style="dim")
    meta_table.add_column(style="cyan")
    meta_table.add_row("Query type", result.query_context.query_type)
    meta_table.add_row(
        "Tickers detected", ", ".join(result.query_context.tickers) or "none"
    )
    meta_table.add_row(
        "Year range",
        (
            f"{result.query_context.year_range[0]}–{result.query_context.year_range[1]}"
            if result.query_context.year_range
            else "all years"
        ),
    )
    meta_table.add_row("Chunks retrieved", str(len(result.chunks)))
    meta_table.add_row("Latency", f"{result.latency_ms}ms")
    console.print(meta_table)

    # ── Retrieved chunks ──────────────────────────────────────────────
    if show_chunks or chunks_only:
        console.print()
        chunk_table = Table(
            "Rank",
            "Header",
            "Score",
            "Method",
            "Preview",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold yellow",
            expand=True,
        )
        for i, retrieved_chunk in enumerate(result.chunks, 1):
            preview = retrieved_chunk.chunk.text[:120].replace("\n", " ")
            chunk_table.add_row(
                str(i),
                retrieved_chunk.chunk.provenance_header(),
                f"{retrieved_chunk.score:.3f}",
                retrieved_chunk.retrieval_method,
                preview,
            )
        console.print(chunk_table)

    if chunks_only:
        return

    # ── Answer ────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            Markdown(result.answer),
            title="[bold yellow]Answer[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def interactive_loop() -> None:
    from src.pipeline import build_pipeline

    pipeline = build_pipeline()
    if pipeline._store.count() == 0:
        console.print(
            "[red]✗ Index is empty. Run: uv run python scripts/build_index.py[/red]"
        )
        sys.exit(1)

    console.print(
        Panel(
            "[bold]SEC Filing Intelligence[/bold] — Interactive Mode\n"
            "[dim]Commands: 'chunks' to toggle chunk display, 'quit' to exit[/dim]",
            border_style="yellow",
        )
    )

    show_chunks = False
    while True:
        try:
            question = console.input("\n[yellow]▶[/yellow] Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "chunks":
            show_chunks = not show_chunks
            console.print(f"[dim]Chunk display: {'ON' if show_chunks else 'OFF'}[/dim]")
            continue

        run_query(question, show_chunks=show_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SEC filing RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--show-chunks", action="store_true", help="Display retrieved chunks"
    )
    parser.add_argument(
        "--chunks-only", action="store_true", help="Show chunks only, skip LLM call"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive REPL mode"
    )
    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_loop()
    else:
        run_query(
            args.question, show_chunks=args.show_chunks, chunks_only=args.chunks_only
        )


if __name__ == "__main__":
    main()
