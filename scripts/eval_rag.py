#!/usr/bin/env python3
"""
Systematic evaluation script for the SEC filing RAG pipeline.

Runs 10 test queries through the full pipeline, scores each one across 5 dimensions
using an LLM-as-judge, and computes automated metrics. Outputs a Rich terminal summary
table, a detailed Markdown report, and a JSON scores file.

Usage:
    uv run python scripts/eval_rag.py
    uv run python scripts/eval_rag.py --cases TC01,TC03,TC09
    uv run python scripts/eval_rag.py --no-judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.llm import LiteLLMClient
from src.pipeline import build_pipeline

console = Console()

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "TC01",
        "query": "What are Apple's primary risk factors in their most recent 10-K?",
        "query_type": "general",
        "expected_tickers": ["AAPL"],
        "expected_sections": ["Item 1A"],
        "notes": "Should cite specific risks with AAPL citations",
    },
    {
        "id": "TC02",
        "query": "How has NVIDIA's revenue grown from 2022 to 2024?",
        "query_type": "trend",
        "expected_tickers": ["NVDA"],
        "expected_sections": ["Item 7", "Item 8"],
        "notes": "Should present data chronologically with specific figures",
    },
    {
        "id": "TC03",
        "query": "Compare Apple and Microsoft on their cloud and services revenue strategy",
        "query_type": "comparison",
        "expected_tickers": ["AAPL", "MSFT"],
        "expected_sections": ["Item 1", "Item 7"],
        "notes": "Should have dedicated sections per company plus comparative summary",
    },
    {
        "id": "TC04",
        "query": "What cybersecurity risks do major tech companies face according to their SEC filings?",
        "query_type": "thematic",
        "expected_tickers": [],
        "expected_sections": ["Item 1A"],
        "notes": "Should group by theme not company; cover multiple companies",
    },
    {
        "id": "TC05",
        "query": "What was Apple's total net sales in fiscal year 2024?",
        "query_type": "general",
        "expected_tickers": ["AAPL"],
        "expected_sections": ["Item 7", "Item 8"],
        "notes": "Specific numeric question — should cite exact figure or state not available",
    },
    {
        "id": "TC06",
        "query": "What legal proceedings and litigation risks does Tesla disclose in recent filings?",
        "query_type": "general",
        "expected_tickers": ["TSLA"],
        "expected_sections": ["Item 3", "Item 1A"],
        "notes": "Should cite Item 3 and any Item 1A legal risk disclosures",
    },
    {
        "id": "TC07",
        "query": "Compare JPMorgan and Goldman Sachs on credit risk and market risk exposure",
        "query_type": "comparison",
        "expected_tickers": ["JPM", "GS"],
        "expected_sections": ["Item 7A", "Item 7", "Item 1A"],
        "notes": "Financial sector comparison; should cite quantitative market risk data",
    },
    {
        "id": "TC08",
        "query": "What do pharmaceutical companies say about drug approval risks and regulatory hurdles?",
        "query_type": "thematic",
        "expected_tickers": [],
        "expected_sections": ["Item 1A", "Item 1"],
        "notes": "Thematic across PFE, ABBV, MRK, JNJ; should not hallucinate drug names",
    },
    {
        "id": "TC09",
        "query": "What did Apple say about their quantum computing research and development strategy?",
        "query_type": "general",
        "expected_tickers": ["AAPL"],
        "expected_sections": [],
        "notes": "Boundary test: this topic is almost certainly not in SEC filings; LLM should say so",
    },
    {
        "id": "TC10",
        "query": "How has Amazon's operating income and margin changed from 2022 to 2024, and what drove the changes?",
        "query_type": "trend",
        "expected_tickers": ["AMZN"],
        "expected_sections": ["Item 7"],
        "notes": "Multi-year trend with causal explanation; should cite figures per year",
    },
]

# ---------------------------------------------------------------------------
# LLM-as-judge configuration
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
# Citation regex helpers
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(
    r"\[([A-Z.]+)\s*\|\s*(10-[KQ])\s*\|\s*(\d{4}-\d{2})\s*\|\s*(Item\s+\w+)\]"
)

_CITATION_RAW_RE = re.compile(r"\[[\w.]+\s*\|\s*10-[KQ]\s*\|\s*\d{4}")


def extract_citations(answer: str) -> list[tuple[str, str, str, str]]:
    """Return list of (ticker, form, period, section) tuples from the answer."""
    return _CITATION_RE.findall(answer)


def count_citations(answer: str) -> int:
    return len(_CITATION_RAW_RE.findall(answer))


def sentences_with_numbers_uncited(answer: str) -> int:
    """Count sentences containing digits or % that do NOT have a citation on the same line."""
    count = 0
    for sentence in answer.split(". "):
        has_number = bool(re.search(r"\d|%", sentence))
        has_citation = "[" in sentence
        if has_number and not has_citation:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Automated metrics
# ---------------------------------------------------------------------------

def compute_automated_metrics(
    answer: str,
    chunks: list,
    expected_tickers: list[str],
    expected_sections: list[str],
) -> dict:
    chunk_sections = [rc.chunk.section for rc in chunks]
    chunk_tickers = [rc.chunk.metadata.ticker for rc in chunks]
    chunk_scores = [rc.score for rc in chunks]

    section_diversity = len(set(chunk_sections))
    ticker_diversity = len(set(chunk_tickers))
    avg_score = mean(chunk_scores) if chunk_scores else 0.0

    top5_sections = [rc.chunk.section for rc in chunks[:5]]
    top_chunk_section = Counter(top5_sections).most_common(1)[0][0] if top5_sections else "N/A"

    answer_words = len(answer.split())
    citation_count = count_citations(answer)
    uncited_number_sentences = sentences_with_numbers_uncited(answer)

    # Expected ticker coverage: fraction of expected_tickers found in retrieved chunks
    retrieved_tickers = set(chunk_tickers)
    if expected_tickers:
        ticker_coverage = sum(1 for t in expected_tickers if t in retrieved_tickers) / len(expected_tickers)
    else:
        ticker_coverage = 1.0  # no expectation → full coverage by default

    # Expected section coverage: check if any retrieved chunk section contains the expected section prefix
    retrieved_sections_lower = {s.lower() for s in chunk_sections}
    if expected_sections:
        section_hits = 0
        for exp_sec in expected_sections:
            exp_lower = exp_sec.lower()
            if any(exp_lower in rs for rs in retrieved_sections_lower):
                section_hits += 1
        section_coverage = section_hits / len(expected_sections)
    else:
        section_coverage = 1.0

    return {
        "citation_count": citation_count,
        "sentences_with_numbers_uncited": uncited_number_sentences,
        "chunk_section_diversity": section_diversity,
        "chunk_ticker_diversity": ticker_diversity,
        "avg_chunk_score": round(avg_score, 5),
        "top_chunk_section": top_chunk_section,
        "answer_length_words": answer_words,
        "expected_ticker_coverage": round(ticker_coverage, 3),
        "expected_section_coverage": round(section_coverage, 3),
    }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def build_judge_prompt(question: str, chunks: list, answer: str) -> str:
    excerpt_parts: list[str] = []
    for i, rc in enumerate(chunks, 1):
        header = rc.chunk.provenance_header()
        excerpt_parts.append(f"[Excerpt {i}] {header}\n{rc.chunk.text}")
    excerpts_block = "\n\n".join(excerpt_parts)

    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EXCERPTS:\n{'—' * 60}\n{excerpts_block}\n{'—' * 60}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Please evaluate the answer and return a JSON object with the required fields."
    )


def run_judge(
    llm: LiteLLMClient,
    question: str,
    chunks: list,
    answer: str,
) -> dict | None:
    prompt = build_judge_prompt(question, chunks, answer)
    try:
        raw = llm.complete(prompt, system=JUDGE_SYSTEM)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[yellow]  Warning: judge JSON parse failed ({exc}). Storing raw text.[/yellow]")
        return {"parse_error": str(exc), "raw_response": raw[:500]}
    except Exception as exc:
        console.print(f"[red]  Warning: judge call failed: {exc}[/red]")
        return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 4.5:
        return "green"
    if score >= 3.5:
        return "yellow"
    if score >= 2.5:
        return "orange1"
    return "red"


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "[dim]N/A[/dim]"
    color = _score_color(score)
    return f"[{color}]{score:.1f}[/{color}]"


def build_terminal_table(results: list[dict]) -> Table:
    table = Table(
        title="[bold]RAG Evaluation Results[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("ID", style="bold", width=5, no_wrap=True)
    table.add_column("Query", min_width=24, max_width=42, no_wrap=False)
    table.add_column("Faith", justify="center", width=6, no_wrap=True)
    table.add_column("Relev", justify="center", width=6, no_wrap=True)
    table.add_column("Compl", justify="center", width=6, no_wrap=True)
    table.add_column("Specif", justify="center", width=6, no_wrap=True)
    table.add_column("Cite", justify="center", width=6, no_wrap=True)
    table.add_column("Overall", justify="center", width=8, no_wrap=True)
    table.add_column("Cites", justify="center", width=5, no_wrap=True)
    table.add_column("Words", justify="center", width=6, no_wrap=True)
    table.add_column("Latency", justify="right", width=9, no_wrap=True)

    for r in results:
        j = r.get("judge_scores") or {}
        auto = r["automated_metrics"]

        table.add_row(
            r["id"],
            r["query"][:38] + "…" if len(r["query"]) > 40 else r["query"],
            _fmt_score(j.get("faithfulness")),
            _fmt_score(j.get("relevance")),
            _fmt_score(j.get("completeness")),
            _fmt_score(j.get("specificity")),
            _fmt_score(j.get("citation_quality")),
            _fmt_score(j.get("overall_score")),
            str(auto["citation_count"]),
            str(auto["answer_length_words"]),
            f"{r['latency_ms']:.0f}ms",
        )

    return table


def write_markdown_report(results: list[dict], report_path: Path) -> None:
    lines: list[str] = [
        "# RAG Pipeline Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Test cases run:** {len(results)}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| ID | Query | Faithfulness | Relevance | Completeness | Specificity | Citation Quality | Overall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r.get("judge_scores") or {}

        def _fmt(k: str) -> str:
            v = j.get(k)
            return str(v) if v is not None else "N/A"

        short_q = r["query"][:50] + "…" if len(r["query"]) > 50 else r["query"]
        lines.append(
            f"| {r['id']} | {short_q} | {_fmt('faithfulness')} | "
            f"{_fmt('relevance')} | {_fmt('completeness')} | "
            f"{_fmt('specificity')} | {_fmt('citation_quality')} | "
            f"{_fmt('overall_score')} |"
        )

    lines += ["", "---", ""]

    # Per-test-case detail
    for r in results:
        tc = r["test_case"]
        j = r.get("judge_scores") or {}
        auto = r["automated_metrics"]

        lines += [
            f"## {r['id']}: {tc['query']}",
            "",
            "### Query Metadata",
            "",
            f"- **Expected query type:** {tc['query_type']}",
            f"- **Expected tickers:** {', '.join(tc['expected_tickers']) or 'any'}",
            f"- **Expected sections:** {', '.join(tc['expected_sections']) or 'any'}",
            f"- **Notes:** {tc['notes']}",
            f"- **Latency:** {r['latency_ms']:.1f} ms",
            "",
            "### Query Analysis",
            "",
        ]
        ctx = r["query_context"]
        lines += [
            f"- **Detected query type:** {ctx['query_type']}",
            f"- **Detected tickers:** {', '.join(ctx['tickers']) or 'none'}",
            f"- **Year range:** {ctx['year_range'] or 'all years'}",
            f"- **Section hints:** {', '.join(ctx['section_hints']) or 'none'}",
            f"- **Sub-queries:** {'; '.join(ctx['sub_queries'])}",
            "",
            "### Retrieved Chunks",
            "",
            "| Rank | Provenance | Score | Method |",
            "|---|---|---|---|",
        ]
        for i, chunk_info in enumerate(r["chunks"], 1):
            lines.append(
                f"| {i} | {chunk_info['provenance']} | "
                f"{chunk_info['score']:.5f} | {chunk_info['retrieval_method']} |"
            )

        lines += [
            "",
            "### Answer",
            "",
            r["answer"],
            "",
            "### Automated Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in auto.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

        if j and "parse_error" not in j:
            lines += [
                "",
                "### Judge Scores",
                "",
                "| Dimension | Score | Reasoning |",
                "|---|---|---|",
            ]
            for dim in ["faithfulness", "relevance", "completeness", "specificity", "citation_quality"]:
                score = j.get(dim, "N/A")
                reasoning = j.get(f"{dim}_reasoning", "")
                lines.append(f"| {dim.replace('_', ' ').title()} | {score} | {reasoning} |")

            lines += [
                "",
                f"**Overall Score:** {j.get('overall_score', 'N/A')}",
                "",
            ]

            hallucinations = j.get("hallucinations", [])
            if hallucinations:
                lines += ["**Hallucinations flagged:**", ""]
                for h in hallucinations:
                    lines.append(f"- {h}")
                lines.append("")

            missed = j.get("missed_key_content", [])
            if missed:
                lines += ["**Missed key content:**", ""]
                for m in missed:
                    lines.append(f"- {m}")
                lines.append("")
        elif j and "parse_error" in j:
            lines += [
                "",
                f"*Judge JSON parse failed: {j['parse_error']}*",
                "",
            ]
        else:
            lines += ["", "*Judge evaluation skipped or failed.*", ""]

        lines += ["---", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_scores_json(results: list[dict], scores_path: Path) -> None:
    output = []
    for r in results:
        j = r.get("judge_scores") or {}
        auto = r["automated_metrics"]
        entry = {
            "id": r["id"],
            "query": r["query"],
            "latency_ms": round(r["latency_ms"], 1),
            "automated": auto,
        }
        if j and "parse_error" not in j:
            entry["judge"] = {
                k: j[k]
                for k in [
                    "faithfulness", "relevance", "completeness",
                    "specificity", "citation_quality", "overall_score",
                ]
                if k in j
            }
        else:
            entry["judge"] = None
        output.append(entry)

    scores_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(case_ids: list[str] | None, use_judge: bool) -> None:
    # Build pipeline
    console.print(Panel("[bold]SEC Filing RAG — Evaluation[/bold]", border_style="cyan"))

    with console.status("[dim]Building pipeline…[/dim]"):
        pipeline = build_pipeline()

    if not pipeline.is_indexed():
        console.print("[red]Index is empty. Run: uv run python scripts/build_index.py[/red]")
        sys.exit(1)

    total_chunks = pipeline._store.count()
    console.print(f"[dim]Index ready — {total_chunks:,} chunks[/dim]\n")

    # Filter test cases
    cases = TEST_CASES
    if case_ids:
        id_set = {cid.strip().upper() for cid in case_ids}
        cases = [tc for tc in TEST_CASES if tc["id"] in id_set]
        if not cases:
            console.print(f"[red]No matching test cases for: {case_ids}[/red]")
            sys.exit(1)
        console.print(f"[dim]Running {len(cases)} of {len(TEST_CASES)} test cases[/dim]\n")

    judge_llm = LiteLLMClient() if use_judge else None

    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluations…", total=len(cases))

        for tc in cases:
            progress.update(task, description=f"[cyan]{tc['id']}[/cyan] {tc['query'][:55]}…")

            # Run query
            try:
                result = pipeline.query(tc["query"])
            except Exception as exc:
                console.print(f"[red]  {tc['id']}: query failed — {exc}[/red]")
                progress.advance(task)
                continue

            answer = result.answer
            chunks = result.chunks
            ctx = result.query_context

            # Automated metrics
            auto_metrics = compute_automated_metrics(
                answer, chunks, tc["expected_tickers"], tc["expected_sections"]
            )

            # LLM judge
            judge_scores: dict | None = None
            if use_judge and judge_llm is not None:
                judge_scores = run_judge(judge_llm, tc["query"], chunks, answer)

            # Serialize chunk info for reporting
            chunk_infos = [
                {
                    "provenance": rc.chunk.provenance_header(),
                    "section": rc.chunk.section,
                    "ticker": rc.chunk.metadata.ticker,
                    "score": rc.score,
                    "retrieval_method": rc.retrieval_method,
                    "text_preview": rc.chunk.text[:120].replace("\n", " "),
                }
                for rc in chunks
            ]

            results.append(
                {
                    "id": tc["id"],
                    "query": tc["query"],
                    "test_case": tc,
                    "answer": answer,
                    "latency_ms": result.latency_ms,
                    "query_context": {
                        "query_type": ctx.query_type,
                        "tickers": ctx.tickers,
                        "year_range": ctx.year_range,
                        "section_hints": ctx.section_hints,
                        "sub_queries": ctx.sub_queries,
                    },
                    "chunks": chunk_infos,
                    "automated_metrics": auto_metrics,
                    "judge_scores": judge_scores,
                }
            )

            progress.advance(task)

    # Print summary table
    console.print()
    console.print(build_terminal_table(results))

    # Aggregate judge scores if available
    judge_results = [r for r in results if r.get("judge_scores") and "parse_error" not in (r["judge_scores"] or {})]
    if judge_results:
        dims = ["faithfulness", "relevance", "completeness", "specificity", "citation_quality", "overall_score"]
        console.print()
        agg_table = Table(title="Aggregate Judge Scores", box=box.SIMPLE_HEAVY, header_style="bold magenta")
        agg_table.add_column("Dimension")
        agg_table.add_column("Mean", justify="right")
        agg_table.add_column("Min", justify="right")
        agg_table.add_column("Max", justify="right")
        for dim in dims:
            vals = [r["judge_scores"][dim] for r in judge_results if dim in r["judge_scores"]]
            if vals:
                agg_table.add_row(
                    dim.replace("_", " ").title(),
                    _fmt_score(mean(vals)),
                    _fmt_score(min(vals)),
                    _fmt_score(max(vals)),
                )
        console.print(agg_table)

    # Write report files
    reports_dir = Path(__file__).parent.parent / "reports" / "eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"{ts}_eval_report.md"
    scores_path = reports_dir / f"{ts}_eval_scores.json"

    write_markdown_report(results, report_path)
    write_scores_json(results, scores_path)

    console.print()
    console.print(f"[green]Report saved  →[/green] {report_path}")
    console.print(f"[green]Scores saved  →[/green] {scores_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SEC filing RAG pipeline across 10 test cases."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated test case IDs to run, e.g. TC01,TC03,TC09",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        default=False,
        help="Skip LLM-as-judge evaluation; run automated metrics only",
    )
    args = parser.parse_args()

    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
    run_eval(case_ids=case_ids, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
