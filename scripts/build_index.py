#!/usr/bin/env python
"""
Build (or rebuild) the vector index from the SEC filing corpus.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --corpus-dir edgar_corpus/ --force
    python scripts/build_index.py --embedder local --vector-store chroma
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

load_dotenv()

console = Console()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides before importing pipeline (env vars read at import)
    if args.embedder:
        os.environ["EMBEDDER"] = args.embedder
    if args.vector_store:
        os.environ["VECTOR_STORE"] = args.vector_store

    from src.pipeline import build_pipeline

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
    console.rule()

    pipeline = build_pipeline()

    progress_state: dict = {"current": 0, "filename": ""}

    def progress_callback(current: int, total: int, filename: str) -> None:
        progress_state["current"] = current
        progress_state["filename"] = filename

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} files"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Indexing filings…", total=len(txt_files))

        t0 = time.perf_counter()
        result = pipeline.index_corpus(
            corpus_dir=corpus_dir,
            force=args.force,
            progress_callback=lambda cur, tot, name: (
                progress.update(task, completed=cur, description=f"[dim]{name}[/dim]")
            ),
        )
        elapsed = time.perf_counter() - t0

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  Files processed : [green]{result['files_processed']}[/green]")
    console.print(f"  Files skipped   : {result['files_skipped']} (already indexed)")
    console.print(f"  Chunks added    : [green]{result['chunks_added']:,}[/green]")
    console.print(f"  Total in index  : [cyan]{result['total_indexed']:,}[/cyan]")
    console.print(f"  Time elapsed    : {elapsed:.1f}s")
    console.rule()


if __name__ == "__main__":
    main()
