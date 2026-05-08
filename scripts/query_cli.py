#!/usr/bin/env python3
"""
Terminal query interface for the SEC RAG pipeline.

Usage:
    uv run python scripts/query_cli.py "What are Apple's risk factors?"
    uv run python scripts/query_cli.py --show-chunks "How has NVIDIA's revenue changed?"
    uv run python scripts/query_cli.py --chunks-only "Compare Apple and Tesla risks"
    uv run python scripts/query_cli.py --profile "What are Apple's risk factors?"
    uv run python scripts/query_cli.py --interactive
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

_ANSWER_OUTPUTS_DIR = Path(__file__).parent.parent / "answer_outputs"


def _sanitize_filename(query: str, max_len: int = 60) -> str:
    """Convert a query string to a safe filename fragment."""
    slug = query.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len]


def _save_output(question: str, result, show_chunks: bool) -> Path:
    """Write answer + metadata to a timestamped Markdown file."""
    _ANSWER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_filename(question)
    path = _ANSWER_OUTPUTS_DIR / f"{ts}_{slug}.md"

    ctx = result.query_context
    year_range_str = (
        f"{ctx.year_range[0]}–{ctx.year_range[1]}"
        if ctx.year_range
        else "all years"
    )
    tickers_str = ", ".join(ctx.tickers) or "none"

    lines = [
        f"# Query: {question}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Query type | {ctx.query_type} |",
        f"| Tickers detected | {tickers_str} |",
        f"| Year range | {year_range_str} |",
        f"| Section hints | {', '.join(ctx.section_hints) or 'none'} |",
        f"| Chunks retrieved | {len(result.chunks)} |",
        f"| Latency | {result.latency_ms:.1f} ms |",
    ]

    timing = result.metadata.get("timing", {})
    if timing:
        lines += [
            "",
            "### Substep Timing",
            "",
            "| Substep | Time (ms) | % of total |",
            "|---|---|---|",
        ]
        total_timed = sum(timing.values())
        for substep, ms in sorted(timing.items(), key=lambda x: -x[1]):
            pct = 100 * ms / result.latency_ms if result.latency_ms else 0
            lines.append(f"| {substep} | {ms:.1f} | {pct:.1f}% |")

    if ctx.sub_queries and len(ctx.sub_queries) > 1:
        lines += ["", "### Sub-queries", ""]
        for sq in ctx.sub_queries:
            lines.append(f"- {sq}")

    if show_chunks and result.chunks:
        lines += [
            "",
            "## Retrieved Chunks",
            "",
            "| Rank | Header | Score | Method |",
            "|---|---|---|---|",
        ]
        for i, rc in enumerate(result.chunks, 1):
            lines.append(
                f"| {i} | {rc.chunk.provenance_header()} "
                f"| {rc.score:.3f} | {rc.retrieval_method} |"
            )

    lines += [
        "",
        "## Answer",
        "",
        result.answer,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_meta(question: str, context, chunks: list, latency_ms: float) -> None:
    meta_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta_table.add_column(style="dim")
    meta_table.add_column(style="cyan")
    meta_table.add_row("Query type", context.query_type)
    meta_table.add_row("Tickers detected", ", ".join(context.tickers) or "none")
    meta_table.add_row(
        "Year range",
        (
            f"{context.year_range[0]}–{context.year_range[1]}"
            if context.year_range
            else "all years"
        ),
    )
    meta_table.add_row("Chunks retrieved", str(len(chunks)))
    meta_table.add_row("Latency", f"{latency_ms:.1f}ms")
    console.print(meta_table)


def _print_chunks(chunks: list) -> None:
    chunk_table = Table(
        "Rank", "Header", "Score", "Method", "Preview",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
        expand=True,
    )
    for i, rc in enumerate(chunks, 1):
        preview = rc.chunk.text[:120].replace("\n", " ")
        chunk_table.add_row(
            str(i),
            rc.chunk.provenance_header(),
            f"{rc.score:.3f}",
            rc.retrieval_method,
            preview,
        )
    console.print(chunk_table)


def run_query(
    question: str,
    show_chunks: bool = False,
    chunks_only: bool = False,
    profile: bool = False,
    no_stream: bool = False,
) -> None:
    from src.pipeline import build_pipeline
    from src.pipeline.rag_pipeline import QueryResult

    pipeline = build_pipeline()

    if pipeline._store.count() == 0:
        console.print(
            "[red]✗ Index is empty. Run: uv run python scripts/build_index.py[/red]"
        )
        sys.exit(1)

    # ── Non-streaming path (used with --profile or --no-stream) ──────
    if no_stream or profile:
        with console.status("[dim]Analyzing query…[/dim]"):
            result = pipeline.query(question, profile=profile)

        _print_meta(question, result.query_context, result.chunks, result.latency_ms)

        if profile:
            timing = result.metadata.get("timing", {})
            if timing:
                prof_table = Table(
                    "Substep", "Time (ms)", "% of total",
                    box=box.SIMPLE_HEAVY,
                    show_header=True,
                    header_style="bold magenta",
                    title="[bold]Latency Breakdown[/bold]",
                )
                for substep, ms in sorted(timing.items(), key=lambda x: -x[1]):
                    pct = 100 * ms / result.latency_ms if result.latency_ms else 0
                    color = "red" if pct > 70 else ("yellow" if pct > 30 else "green")
                    prof_table.add_row(
                        substep,
                        f"[{color}]{ms:.1f}[/{color}]",
                        f"[{color}]{pct:.1f}%[/{color}]",
                    )
                unaccounted = result.latency_ms - sum(timing.values())
                prof_table.add_row(
                    "[dim]overhead/other[/dim]",
                    f"[dim]{unaccounted:.1f}[/dim]",
                    f"[dim]{100 * unaccounted / result.latency_ms:.1f}%[/dim]",
                )
                console.print()
                console.print(prof_table)

        if show_chunks or chunks_only:
            console.print()
            _print_chunks(result.chunks)

        if chunks_only:
            return

        console.print()
        console.print(
            Panel(
                Markdown(result.answer),
                title="[bold yellow]Answer[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        output_path = _save_output(question, result, show_chunks)
        console.print(f"\n[dim]Output saved → {output_path}[/dim]")
        return

    # ── Streaming path (default) ──────────────────────────────────────
    with console.status("[dim]Retrieving context…[/dim]"):
        context, chunks, token_stream, retrieval_ms = pipeline.stream_query(question)

    _print_meta(question, context, chunks, retrieval_ms)

    if show_chunks or chunks_only:
        console.print()
        _print_chunks(chunks)

    if chunks_only:
        return

    # Stream the answer token by token inside a Live panel
    console.print()
    answer_parts: list[str] = []
    t_llm_start = time.perf_counter()

    with Live(
        Panel("", title="[bold yellow]Answer[/bold yellow]", border_style="yellow", padding=(1, 2)),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:
        for token in token_stream:
            answer_parts.append(token)
            accumulated = "".join(answer_parts)
            live.update(
                Panel(
                    Markdown(accumulated),
                    title="[bold yellow]Answer[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

    llm_ms = (time.perf_counter() - t_llm_start) * 1000
    total_ms = retrieval_ms + llm_ms
    full_answer = "".join(answer_parts)

    console.print(f"[dim]  ↳ retrieval {retrieval_ms:.0f}ms · llm {llm_ms:.0f}ms · total {total_ms:.0f}ms[/dim]")

    # Build a QueryResult-compatible object for the save helper
    class _StreamResult:
        answer = full_answer
        query_context = context
        chunks = chunks
        latency_ms = total_ms
        metadata: dict = {}

    output_path = _save_output(question, _StreamResult(), show_chunks)
    console.print(f"[dim]Output saved → {output_path}[/dim]")


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

        run_query(question, show_chunks=show_chunks, no_stream=False)


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
    parser.add_argument(
        "--profile", action="store_true",
        help="Show per-substep latency breakdown (disables streaming)",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming and wait for the full response before printing",
    )
    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_loop()
    else:
        run_query(
            args.question,
            show_chunks=args.show_chunks,
            chunks_only=args.chunks_only,
            profile=args.profile,
            no_stream=args.no_stream,
        )


if __name__ == "__main__":
    main()
