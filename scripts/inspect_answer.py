#!/usr/bin/env python3
"""
Deep-dive inspection tool for a single SEC RAG pipeline query.

Shows full chunk text, citation audit, coverage gap analysis, unused chunks,
and LLM judge scores — all in a Rich terminal layout.

Usage:
    uv run python scripts/inspect_answer.py "What are Apple's key risk factors?"
    uv run python scripts/inspect_answer.py --no-judge "What are Apple's key risk factors?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.llm import LiteLLMClient
from src.pipeline import build_pipeline

console = Console()

# ---------------------------------------------------------------------------
# LLM judge (same config as eval_rag.py)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert evaluator of RAG (retrieval-augmented generation) systems applied to SEC financial filings.
You will be given a question, the retrieved source excerpts, and an answer generated from those excerpts.
Evaluate the answer strictly and return a JSON object with these exact fields:

{
  "faithfulness": <1-5>,        // Every factual claim traces to a provided excerpt. 5=all claims sourced, 1=significant hallucination
  "faithfulness_reasoning": "<1-2 sentences>",
  "relevance": <1-5>,           // Answer directly addresses the question asked. 5=fully addresses, 1=off-topic
  "relevance_reasoning": "<1-2 sentences>",
  "completeness": <1-5>,        // Answer covers all important aspects visible in the excerpts. 5=comprehensive, 1=major gaps
  "completeness_reasoning": "<1-2 sentences>",
  "specificity": <1-5>,         // Answer cites specific figures, dates, names rather than vague generalities. 5=highly specific, 1=entirely vague
  "specificity_reasoning": "<1-2 sentences>",
  "citation_quality": <1-5>,    // Citations use [TICKER | FORM | PERIOD | SECTION] format consistently. 5=all claims cited correctly, 1=no citations
  "citation_quality_reasoning": "<1-2 sentences>",
  "hallucinations": ["<claim not found in any excerpt>", ...],  // List any claims in the answer not supported by excerpts
  "missed_key_content": ["<important fact from excerpts not in answer>", ...],  // Content in excerpts that should have been included
  "overall_score": <1-5>        // Holistic quality score
}

Be strict. A score of 5 means genuinely excellent, not just acceptable. Score 3 is average/acceptable."""


# ---------------------------------------------------------------------------
# Citation helpers (identical logic to eval_rag.py for consistency)
# ---------------------------------------------------------------------------

_CITATION_FULL_RE = re.compile(
    r"\[([A-Z.]+)\s*\|\s*(10-[KQ])\s*\|\s*(\d{4}-\d{2})\s*\|\s*(Item\s+\w+)\]"
)

_CITATION_RAW_RE = re.compile(r"\[[\w.]+\s*\|\s*10-[KQ]\s*\|\s*\d{4}")


def extract_citations(answer: str) -> list[tuple[str, str, str, str]]:
    """Return list of (ticker, form, period, section) tuples."""
    return _CITATION_FULL_RE.findall(answer)


def sentences_with_numbers_uncited(answer: str) -> list[str]:
    """Return sentences that contain digits or % but no citation bracket."""
    flagged: list[str] = []
    for sentence in answer.split(". "):
        has_number = bool(re.search(r"\d|%", sentence))
        has_citation = "[" in sentence
        if has_number and not has_citation:
            flagged.append(sentence.strip())
    return flagged


# ---------------------------------------------------------------------------
# Unused chunk detection
# ---------------------------------------------------------------------------

def _extract_key_noun_phrases(text: str, top_n: int = 8) -> list[str]:
    """
    Heuristic: extract the most distinctive multi-word sequences from chunk text.
    Uses quoted strings, capitalized phrases, and numbers as signal.
    """
    phrases: list[str] = []

    # Quoted strings are high-signal
    phrases.extend(re.findall(r'"([^"]{6,60})"', text))

    # Numbers with context (e.g. "$3.5 billion", "42%")
    phrases.extend(re.findall(r"\$[\d,.]+\s*(?:billion|million|thousand)?", text, re.IGNORECASE))
    phrases.extend(re.findall(r"\d+\.?\d*\s*%", text))
    phrases.extend(re.findall(r"\b\d[\d,.]*\s*(?:billion|million|thousand)\b", text, re.IGNORECASE))

    # Proper-noun-like capitalized phrases (2+ consecutive Title-case words)
    phrases.extend(re.findall(r"\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b", text))

    # Deduplicate and return top_n
    seen: set[str] = set()
    unique: list[str] = []
    for p in phrases:
        p = p.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)

    return unique[:top_n]


def find_unused_chunks(answer: str, chunks: list) -> list[tuple[int, str, list[str]]]:
    """
    Return (rank, provenance, key_phrases) for chunks whose key phrases
    don't appear in the answer.
    """
    answer_lower = answer.lower()
    unused: list[tuple[int, str, list[str]]] = []

    for i, rc in enumerate(chunks, 1):
        key_phrases = _extract_key_noun_phrases(rc.chunk.text)
        if not key_phrases:
            continue

        # A chunk is considered "used" if at least one key phrase appears in the answer
        matched = any(phrase.lower() in answer_lower for phrase in key_phrases)
        if not matched:
            unused.append((i, rc.chunk.provenance_header(), key_phrases))

    return unused


# ---------------------------------------------------------------------------
# Citation audit
# ---------------------------------------------------------------------------

def audit_citations(
    answer: str, chunks: list
) -> list[dict]:
    """
    For each citation found in the answer, check whether it matches a retrieved chunk.
    Returns list of dicts with citation text and match status.
    """
    citations = extract_citations(answer)
    chunk_provenances: list[tuple[str, str, str, str]] = []
    for rc in chunks:
        m = rc.chunk.metadata
        period = m.report_period[:7]  # YYYY-MM
        chunk_provenances.append((
            m.ticker,
            m.filing_type,
            period,
            rc.chunk.section,
        ))

    results: list[dict] = []
    for ticker, form, period, section in citations:
        # Fuzzy match: check if any retrieved chunk has the same ticker + form + period,
        # and whether the section prefix matches (e.g. "Item 1A" in "Item 1A - Risk Factors")
        matched = False
        for c_ticker, c_form, c_period, c_section in chunk_provenances:
            ticker_ok = ticker == c_ticker
            form_ok = form == c_form
            period_ok = period == c_period
            section_ok = section.lower().replace(" ", "") in c_section.lower().replace(" ", "")
            if ticker_ok and form_ok and period_ok and section_ok:
                matched = True
                break

        citation_str = f"[{ticker} | {form} | {period} | {section}]"
        results.append({
            "citation": citation_str,
            "ticker": ticker,
            "form": form,
            "period": period,
            "section": section,
            "matched": matched,
        })

    return results


# ---------------------------------------------------------------------------
# LLM judge call
# ---------------------------------------------------------------------------

def run_judge(llm: LiteLLMClient, question: str, chunks: list, answer: str) -> dict | None:
    excerpt_parts: list[str] = []
    for i, rc in enumerate(chunks, 1):
        header = rc.chunk.provenance_header()
        excerpt_parts.append(f"[Excerpt {i}] {header}\n{rc.chunk.text}")
    excerpts_block = "\n\n".join(excerpt_parts)

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EXCERPTS:\n{'—' * 60}\n{excerpts_block}\n{'—' * 60}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Please evaluate the answer and return a JSON object with the required fields."
    )

    try:
        raw = llm.complete(prompt, system=JUDGE_SYSTEM)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[yellow]Warning: judge JSON parse failed ({exc})[/yellow]")
        return {"parse_error": str(exc), "raw_response": raw[:500]}
    except Exception as exc:
        console.print(f"[red]Warning: judge call failed: {exc}[/red]")
        return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 4.5:
        return "green"
    if score >= 3.5:
        return "yellow"
    if score >= 2.5:
        return "orange1"
    return "red"


def _render_score(score: int | float | None) -> str:
    if score is None:
        return "N/A"
    color = _score_color(float(score))
    return f"[{color}]{score}[/{color}]"


def print_query_analysis(question: str, ctx) -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    table.add_column(style="dim", width=22)
    table.add_column(style="cyan")

    year_str = (
        f"{ctx.year_range[0]}–{ctx.year_range[1]}" if ctx.year_range else "all years"
    )
    table.add_row("Query type", ctx.query_type)
    table.add_row("Detected tickers", ", ".join(ctx.tickers) if ctx.tickers else "[dim]none[/dim]")
    table.add_row("Year range", year_str)
    table.add_row("Section hints", ", ".join(ctx.section_hints) if ctx.section_hints else "[dim]none[/dim]")
    if ctx.sub_queries and len(ctx.sub_queries) > 1:
        for i, sq in enumerate(ctx.sub_queries, 1):
            label = f"Sub-query {i}" if i > 1 else "Sub-queries"
            table.add_row(label, sq)
    else:
        table.add_row("Sub-queries", ctx.sub_queries[0] if ctx.sub_queries else "[dim]none[/dim]")

    console.print(
        Panel(table, title="[bold cyan]Query Analysis[/bold cyan]", border_style="cyan")
    )


def print_retrieved_chunks(chunks: list) -> None:
    console.print(Rule("[bold yellow]Retrieved Chunks[/bold yellow]", style="yellow"))
    for i, rc in enumerate(chunks, 1):
        header = rc.chunk.provenance_header()
        score_color = "green" if rc.score > 0.015 else "yellow" if rc.score > 0.010 else "dim"
        meta_line = (
            f"[bold]Rank {i}[/bold]  [{score_color}]score={rc.score:.5f}[/{score_color}]  "
            f"[dim]method={rc.retrieval_method}[/dim]"
        )
        console.print(
            Panel(
                f"[dim]{rc.chunk.text}[/dim]",
                title=f"[bold yellow]{header}[/bold yellow]  {meta_line}",
                border_style="dim",
                padding=(0, 1),
            )
        )


def print_section_distribution(chunks: list) -> None:
    section_counts: Counter = Counter(rc.chunk.section for rc in chunks)

    table = Table(
        "Section", "Count", "Tickers",
        box=box.SIMPLE_HEAVY,
        title="[bold]Section Distribution[/bold]",
        header_style="bold magenta",
    )
    for section, count in section_counts.most_common():
        tickers_in_section = sorted({
            rc.chunk.metadata.ticker
            for rc in chunks
            if rc.chunk.section == section
        })
        table.add_row(section, str(count), ", ".join(tickers_in_section))

    console.print(table)


def print_answer_with_citation_audit(question: str, answer: str, chunks: list) -> None:
    console.print(
        Panel(
            answer,
            title="[bold yellow]Answer[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # Citation audit
    audit = audit_citations(answer, chunks)
    if not audit:
        console.print("[dim]  No citations found in answer.[/dim]")
        return

    audit_table = Table(
        "Citation", "Matched Chunk?",
        box=box.SIMPLE_HEAVY,
        title="[bold]Citation Audit[/bold]",
        header_style="bold cyan",
    )
    for entry in audit:
        matched_str = "[green]Yes[/green]" if entry["matched"] else "[red]No — not in retrieved set[/red]"
        audit_table.add_row(entry["citation"], matched_str)

    console.print(audit_table)

    # Summary line
    matched_count = sum(1 for e in audit if e["matched"])
    total_count = len(audit)
    if total_count > 0:
        ratio_color = "green" if matched_count == total_count else "yellow" if matched_count > total_count // 2 else "red"
        console.print(
            f"  [{ratio_color}]{matched_count}/{total_count} citations matched retrieved chunks[/{ratio_color}]"
        )


def print_coverage_gaps(answer: str) -> None:
    flagged = sentences_with_numbers_uncited(answer)
    if not flagged:
        console.print(
            Panel(
                "[green]No sentences with uncited numerical claims found.[/green]",
                title="[bold]Coverage Gap Analysis[/bold]",
                border_style="green",
            )
        )
        return

    gap_text = Text()
    for i, sentence in enumerate(flagged, 1):
        gap_text.append(f"{i}. ", style="bold red")
        gap_text.append(sentence + "\n", style="default")

    console.print(
        Panel(
            gap_text,
            title=f"[bold red]Coverage Gaps — {len(flagged)} sentence(s) with uncited numbers[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
    )


def print_unused_chunks(answer: str, chunks: list) -> None:
    unused = find_unused_chunks(answer, chunks)
    if not unused:
        console.print(
            Panel(
                "[green]All retrieved chunks appear to be reflected in the answer.[/green]",
                title="[bold]Unused Chunks[/bold]",
                border_style="green",
            )
        )
        return

    lines: list[str] = []
    for rank, provenance, phrases in unused:
        lines.append(f"[bold]Rank {rank}[/bold] {provenance}")
        lines.append(f"  [dim]Key phrases not found in answer:[/dim] {', '.join(phrases[:5])}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold yellow]Unused Chunks — {len(unused)} chunk(s) not reflected in answer[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def print_judge_scores(judge: dict | None) -> None:
    if judge is None:
        console.print("[dim]Judge evaluation was skipped.[/dim]")
        return

    if "parse_error" in judge:
        console.print(
            Panel(
                f"[red]Judge JSON parse failed:[/red] {judge['parse_error']}\n\n"
                f"[dim]Raw response excerpt:[/dim]\n{judge.get('raw_response', '')}",
                title="[bold red]Judge Evaluation — Parse Error[/bold red]",
                border_style="red",
            )
        )
        return

    dims = [
        ("faithfulness", "Faithfulness"),
        ("relevance", "Relevance"),
        ("completeness", "Completeness"),
        ("specificity", "Specificity"),
        ("citation_quality", "Citation Quality"),
    ]

    score_table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", expand=True)
    score_table.add_column("Dimension", width=20)
    score_table.add_column("Score", justify="center", width=7)
    score_table.add_column("Reasoning")

    for key, label in dims:
        score = judge.get(key)
        reasoning = judge.get(f"{key}_reasoning", "")
        score_table.add_row(label, _render_score(score), reasoning)

    overall = judge.get("overall_score")
    score_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{_render_score(overall)}[/bold]",
        "",
    )

    console.print(
        Panel(
            score_table,
            title="[bold magenta]LLM Judge Scores[/bold magenta]",
            border_style="magenta",
        )
    )

    hallucinations = judge.get("hallucinations", [])
    if hallucinations:
        h_text = "\n".join(f"- {h}" for h in hallucinations)
        console.print(
            Panel(
                f"[red]{h_text}[/red]",
                title="[bold red]Hallucinations Flagged[/bold red]",
                border_style="red",
                padding=(0, 1),
            )
        )

    missed = judge.get("missed_key_content", [])
    if missed:
        m_text = "\n".join(f"- {m}" for m in missed)
        console.print(
            Panel(
                f"[yellow]{m_text}[/yellow]",
                title="[bold yellow]Missed Key Content[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )


# ---------------------------------------------------------------------------
# Main inspection flow
# ---------------------------------------------------------------------------

def inspect(question: str, use_judge: bool) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold]{question}[/bold]",
            title="[bold cyan]SEC RAG — Answer Inspection[/bold cyan]",
            border_style="cyan",
        )
    )

    # Build pipeline and run query
    with console.status("[dim]Building pipeline and running query…[/dim]"):
        pipeline = build_pipeline()

    if not pipeline.is_indexed():
        console.print("[red]Index is empty. Run: uv run python scripts/build_index.py[/red]")
        sys.exit(1)

    with console.status("[dim]Retrieving and generating answer…[/dim]"):
        result = pipeline.query(question, profile=True)

    answer = result.answer
    chunks = result.chunks
    ctx = result.query_context

    latency = result.latency_ms
    timing = result.metadata.get("timing", {})

    # ── 1. Query analysis ────────────────────────────────────────────────
    console.print()
    print_query_analysis(question, ctx)

    if timing:
        timing_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        timing_table.add_column(style="dim", width=25)
        timing_table.add_column(style="cyan", justify="right")
        for substep, ms in sorted(timing.items(), key=lambda x: -x[1]):
            timing_table.add_row(substep, f"{ms:.1f} ms")
        overhead = latency - sum(timing.values())
        timing_table.add_row("[dim]overhead / other[/dim]", f"[dim]{overhead:.1f} ms[/dim]")
        timing_table.add_row("[bold]Total latency[/bold]", f"[bold]{latency:.1f} ms[/bold]")
        console.print(
            Panel(timing_table, title="[bold]Latency Breakdown[/bold]", border_style="dim")
        )

    # ── 2. Retrieved chunks (full text) ──────────────────────────────────
    console.print()
    print_retrieved_chunks(chunks)

    # ── 3. Section distribution ──────────────────────────────────────────
    console.print()
    print_section_distribution(chunks)

    # ── 4. Answer with citation audit ────────────────────────────────────
    console.print()
    print_answer_with_citation_audit(question, answer, chunks)

    # ── 5. Coverage gap analysis ─────────────────────────────────────────
    console.print()
    print_coverage_gaps(answer)

    # ── 6. Unused chunks ─────────────────────────────────────────────────
    console.print()
    print_unused_chunks(answer, chunks)

    # ── 7. LLM judge ─────────────────────────────────────────────────────
    console.print()
    if use_judge:
        with console.status("[dim]Running LLM judge evaluation…[/dim]"):
            judge_llm = LiteLLMClient()
            judge_result = run_judge(judge_llm, question, chunks, answer)
        print_judge_scores(judge_result)
    else:
        console.print("[dim]LLM judge skipped (--no-judge).[/dim]")

    console.print()
    console.print(Rule("[dim]Inspection complete[/dim]", style="dim"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep-dive inspection of a single SEC RAG pipeline query."
    )
    parser.add_argument("question", type=str, help="The question to inspect")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        default=False,
        help="Skip the LLM-as-judge evaluation call",
    )
    args = parser.parse_args()

    inspect(question=args.question, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
