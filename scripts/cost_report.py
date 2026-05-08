"""
Embedding cost report — appends a Markdown entry to COST_REPORT.md each time it runs.

Reads the current vector store to measure chunk/token counts, then estimates
the OpenAI embedding cost at published rates.

Usage:
    uv run python scripts/cost_report.py [--output COST_REPORT.md]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)


# ── Pricing (per 1M tokens, as of 2025) ──────────────────────────────────────
_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
    # local embedders are free
    "BAAI/bge-large-en-v1.5": 0.0,
    "local": 0.0,
}


def _get_store_stats() -> dict:
    """Load the vector store and return chunk + token counts."""
    from src.pipeline import build_pipeline

    pipeline = build_pipeline()
    store = pipeline._store

    if not store.collection_exists():
        return {"error": "No index found. Run scripts/build_index.py first."}

    chunks = store.get_all_chunks()
    total_chunks = len(chunks)
    total_tokens = sum(c.token_count for c in chunks)
    tickers = sorted({c.metadata.ticker for c in chunks})
    filing_types = sorted({c.metadata.filing_type for c in chunks})

    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "tickers": tickers,
        "filing_types": filing_types,
    }


def _build_report_entry(stats: dict, embedder: str, model: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if "error" in stats:
        return f"\n## {ts}\n\n**Error:** {stats['error']}\n"

    total_tokens = stats["total_tokens"]
    total_chunks = stats["total_chunks"]

    price_per_m = _PRICING.get(model, _PRICING.get(embedder, 0.0))
    cost_usd = (total_tokens / 1_000_000) * price_per_m

    tickers_str = ", ".join(stats["tickers"])
    types_str = ", ".join(stats["filing_types"])

    vector_store = os.getenv("VECTOR_STORE", "chroma")
    chroma_path = os.getenv("CHROMA_PATH", "./.chroma")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    store_detail = chroma_path if vector_store == "chroma" else qdrant_url

    lines = [
        f"\n## Index snapshot — {ts}",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Embedder | `{embedder}` |",
        f"| Embedding model | `{model}` |",
        f"| Vector store | `{vector_store}` ({store_detail}) |",
        f"| LLM model | `{os.getenv('LLM_MODEL', 'claude-sonnet-4-6')}` |",
        f"| Total chunks indexed | {total_chunks:,} |",
        f"| Total tokens embedded | {total_tokens:,} |",
        f"| Avg tokens / chunk | {total_tokens // total_chunks if total_chunks else 0:,} |",
        f"| Tickers covered | {len(stats['tickers'])} ({tickers_str}) |",
        f"| Filing types | {types_str} |",
        f"| Embedding cost (est.) | ${cost_usd:.4f} USD @ ${price_per_m}/1M tokens |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Append embedding cost entry to Markdown report")
    ap.add_argument(
        "--output",
        default="COST_REPORT.md",
        help="Path to append-only Markdown report (default: COST_REPORT.md)",
    )
    args = ap.parse_args()

    embedder = os.getenv("EMBEDDER", "openai")
    if embedder == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    else:
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

    print("Reading index stats...")
    stats = _get_store_stats()

    entry = _build_report_entry(stats, embedder, model)

    output_path = Path(args.output)
    if not output_path.exists():
        header = "# Embedding Cost Report\n\nAppend-only log of index snapshots and embedding costs.\n"
        output_path.write_text(header)

    with output_path.open("a") as f:
        f.write(entry)

    print(f"Appended entry to {output_path}")
    if "error" not in stats:
        total_tokens = stats["total_tokens"]
        total_chunks = stats["total_chunks"]
        price_per_m = _PRICING.get(model, 0.0)
        cost = (total_tokens / 1_000_000) * price_per_m
        print(f"  Chunks: {total_chunks:,}  Tokens: {total_tokens:,}  Cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
