#!/usr/bin/env python
"""
Build (or rebuild) the vector index from the SEC filing corpus.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --corpus-dir edgar_corpus/ --force
    uv run python scripts/build_index.py --embedder local --vector-store chroma
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

load_dotenv()


def _setup_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"build_index_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )
    # Keep third-party loggers quieter in the file
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    return log_file


def _write_report(
    reports_dir: Path,
    args: argparse.Namespace,
    result: dict,
    elapsed: float,
    log_file: Path,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_human = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_path = reports_dir / f"build_index_{ts_file}.md"

    embedder = os.getenv("EMBEDDER", "openai")
    store = os.getenv("VECTOR_STORE", "chroma")
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    parse_errors = result.get("parse_errors", [])
    embed_errors = result.get("embed_errors", [])
    chunks_to_embed = result.get("total_to_embed", result["chunks_added"])

    lines = [
        f"# Build Index Report — {ts_human}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Corpus dir | `{args.corpus_dir}` |",
        f"| Mode | {'force re-index' if args.force else 'incremental'} |",
        f"| Embedder | `{embedder}` |",
        f"| Embedding model | `{emb_model}` |",
        f"| Vector store | `{store}` |",
        f"| LLM model | `{model}` |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Files processed (new/updated) | {result['files_processed']} |",
        f"| Files skipped (already indexed) | {result['files_skipped']} |",
        f"| New chunks added | {result['chunks_added']:,} |",
        f"| Total chunks in index | {result['total_indexed']:,} |",
        f"| Chunks sent to embedder | {chunks_to_embed:,} |",
        f"| Time elapsed | {elapsed:.1f}s |",
        f"| Parse errors | {len(parse_errors)} |",
        f"| Embed errors | {len(embed_errors)} |",
        f"| Log file | `{log_file}` |",
        "",
    ]

    if parse_errors:
        lines += ["## Parse Errors", ""]
        for e in parse_errors:
            lines.append(f"- `{e['file']}`: {e['error']}")
        lines.append("")

    if embed_errors:
        lines += ["## Embedding Errors / Dropped Chunks", ""]
        for e in embed_errors:
            if "chunk_id" in e:
                lines.append(
                    f"- Chunk `{e['chunk_id']}` "
                    f"(section: {e['section']}, ~{e['token_count']} tokens)"
                )
            else:
                lines.append(f"- Batch at index {e.get('batch_start', '?')}: {e['error']}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SEC filing RAG index")
    parser.add_argument(
        "--corpus-dir",
        default=os.getenv("CORPUS_DIR", "edgar_corpus"),
        help="Directory containing .txt filing files (default: edgar_corpus/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing index and re-index everything",
    )
    parser.add_argument(
        "--embedder",
        choices=["openai", "local"],
        default=None,
        help="Override EMBEDDER env var",
    )
    parser.add_argument(
        "--vector-store",
        choices=["chroma", "qdrant"],
        default=None,
        dest="vector_store",
        help="Override VECTOR_STORE env var",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for log files (default: logs/)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/build_index_reports",
        help="Directory for build reports (default: reports/build_index_reports/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_file = _setup_logging(Path(args.logs_dir))

    if args.embedder:
        os.environ["EMBEDDER"] = args.embedder
    if args.vector_store:
        os.environ["VECTOR_STORE"] = args.vector_store

    from src.pipeline import build_pipeline

    console = Console()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        console.print(f"[red]Error:[/red] corpus directory not found: {corpus_dir}")
        sys.exit(1)

    txt_files = list(corpus_dir.glob("*.txt"))
    if not txt_files:
        console.print(f"[red]Error:[/red] no .txt files found in {corpus_dir}")
        sys.exit(1)

    console.rule("[bold cyan]SEC Filing Index Builder[/bold cyan]")
    console.print(f"  Corpus  : [cyan]{corpus_dir}[/cyan] ({len(txt_files)} files)")
    console.print(f"  Embedder: [cyan]{os.getenv('EMBEDDER', 'openai')}[/cyan]")
    console.print(f"  Store   : [cyan]{os.getenv('VECTOR_STORE', 'chroma')}[/cyan]")
    console.print(f"  Mode    : [cyan]{'force re-index' if args.force else 'incremental'}[/cyan]")
    console.print(f"  Log     : [dim]{log_file}[/dim]")
    console.rule()

    pipeline = build_pipeline()

    logging.getLogger(__name__).info(
        "Starting index build: %d files, force=%s, embedder=%s, store=%s",
        len(txt_files), args.force,
        os.getenv("EMBEDDER", "openai"), os.getenv("VECTOR_STORE", "chroma"),
    )

    progress_cols = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_cols, console=console, transient=False) as progress:
        parse_task = progress.add_task(
            "[bold]Phase 1[/bold] Parse + chunk", total=len(txt_files)
        )
        # Embed task total is unknown until parsing finishes; start at 0
        embed_task = progress.add_task(
            "[bold]Phase 2[/bold] Embed + store", total=None, visible=False
        )

        embed_total_known = False

        def parse_cb(cur: int, tot: int, name: str) -> None:
            progress.update(
                parse_task,
                completed=cur,
                description=f"[bold]Phase 1[/bold] [dim]{name}[/dim]",
            )

        def embed_cb(cur: int, tot: int, batch_size: int) -> None:
            nonlocal embed_total_known
            if not embed_total_known:
                progress.update(embed_task, total=tot, visible=True)
                embed_total_known = True
            progress.update(
                embed_task,
                completed=cur,
                description=f"[bold]Phase 2[/bold] embedding batch {cur}/{tot}",
            )

        t0 = time.perf_counter()
        result = pipeline.index_corpus(
            corpus_dir=corpus_dir,
            force=args.force,
            parse_progress_callback=parse_cb,
            embed_progress_callback=embed_cb,
        )
        elapsed = time.perf_counter() - t0

        # Ensure both bars show complete
        progress.update(parse_task, completed=len(txt_files))
        if embed_total_known:
            embed_total = result.get("total_to_embed", result["chunks_added"])
            n_batches = max(1, (embed_total + 99) // 100)
            progress.update(embed_task, completed=n_batches)

    parse_errors = result.get("parse_errors", [])
    embed_errors = result.get("embed_errors", [])

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  Files processed   : [green]{result['files_processed']}[/green]")
    console.print(f"  Files skipped     : {result['files_skipped']} (already indexed)")
    console.print(f"  Chunks added      : [green]{result['chunks_added']:,}[/green]")
    console.print(f"  Total in index    : [cyan]{result['total_indexed']:,}[/cyan]")
    console.print(f"  Time elapsed      : {elapsed:.1f}s")
    if parse_errors:
        console.print(f"  Parse errors      : [red]{len(parse_errors)}[/red]")
    if embed_errors:
        console.print(f"  Embed errors      : [yellow]{len(embed_errors)}[/yellow]")

    # Save report
    reports_dir = Path(args.reports_dir)
    report_path = _write_report(reports_dir, args, result, elapsed, log_file)
    console.print(f"  Report saved      : [dim]{report_path}[/dim]")
    console.rule()

    logging.getLogger(__name__).info(
        "Build complete: %d files processed, %d chunks added, %.1fs",
        result["files_processed"], result["chunks_added"], elapsed,
    )


if __name__ == "__main__":
    main()
