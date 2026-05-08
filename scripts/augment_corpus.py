#!/usr/bin/env python3
"""
Augment the local corpus by fetching additional SEC filings via edgartools.

Use this to fill gaps in the provided corpus or fetch more recent filings.
All fetched files are written to the corpus directory in the same format
as the provided .txt files (with the 10-line header block).

Usage:
    # Fetch latest 3 10-K filings for NVIDIA
    uv run python scripts/augment_corpus.py --ticker NVDA --form 10-K --count 3

    # Fetch specific companies matching the assessment corpus tickers
    uv run python scripts/augment_corpus.py --all-corpus-tickers --form 10-K --count 2

    # Dry run (show what would be fetched without downloading)
    uv run python scripts/augment_corpus.py --ticker AAPL --form 10-Q --count 4 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Tickers present in the provided corpus (from manifest inspection)
CORPUS_TICKERS = [
    "AAPL", "AMZN", "BAC", "DIS", "GOOG",
    "JNJ", "KO", "META", "MSFT", "NVDA",
    "PFE", "TSLA", "UNH", "XOM",
]

# Header template — must match the format expected by parser.py
_HEADER_TEMPLATE = """\
Company: {company}
Ticker: {ticker}
Filing Type: {filing_type}
Filing Date: {filing_date}
Report Period: {report_period}
Quarter: {quarter}
CIK: {cik}
URL: {url}
================================================================================
"""


def fetch_filings(
    ticker: str,
    form: str,
    count: int,
    output_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Fetch `count` most recent filings of `form` type for `ticker` via edgartools.

    Returns list of result dicts with status per filing.
    """
    try:
        from edgar import Company, set_identity
    except ImportError:
        console.print("[red]edgartools not installed. Run: uv add edgartools[/red]")
        sys.exit(1)

    set_identity("SEC RAG Pipeline augmentor@example.com")

    results = []
    try:
        company = Company(ticker)
    except Exception as e:
        console.print(f"[red]Could not find company for ticker {ticker}: {e}[/red]")
        return results

    try:
        filings = company.get_filings(form=form)
        recent = filings.latest(count)
        # edgartools returns a single filing or a list
        if not isinstance(recent, list):
            recent = [recent]
    except Exception as e:
        console.print(f"[red]Error fetching filings for {ticker}: {e}[/red]")
        return results

    for filing in recent:
        result = _process_filing(filing, ticker, form, output_dir, dry_run)
        results.append(result)
        time.sleep(0.5)  # be polite to EDGAR rate limits

    return results


def _process_filing(filing, ticker: str, form: str, output_dir: Path, dry_run: bool) -> dict:
    """Extract text from one filing and write to output_dir."""
    try:
        # edgartools filing metadata
        filing_date = str(filing.filing_date)
        period = str(getattr(filing, "period_of_report", filing_date))
        accession = str(filing.accession_no).replace("-", "")
        cik = str(getattr(filing, "cik", ""))
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

        # Derive quarter label
        quarter = _derive_quarter(filing_date, form)

        # Filename convention matching the corpus
        period_tag = period[:7].replace("-", "")  # "202409"
        filename = f"{ticker}_{form.replace('-','')}_{period_tag}_{filing_date}_full.txt"
        output_path = output_dir / filename

        if output_path.exists():
            return {"file": filename, "status": "skipped (exists)", "ticker": ticker}

        if dry_run:
            return {"file": filename, "status": "dry_run", "ticker": ticker}

        # Fetch the actual filing text via edgartools
        obj = filing.obj()
        text = _extract_text(obj)

        if not text or len(text) < 1000:
            return {"file": filename, "status": "error: text too short", "ticker": ticker}

        # Build company name
        company_name = getattr(filing, "company", ticker)

        # Prepend our standard header
        header = _HEADER_TEMPLATE.format(
            company=company_name,
            ticker=ticker,
            filing_type=form,
            filing_date=filing_date,
            report_period=period,
            quarter=quarter,
            cik=cik,
            url=url,
        )

        output_path.write_text(header + text, encoding="utf-8")
        return {
            "file": filename,
            "status": "ok",
            "ticker": ticker,
            "chars": len(text),
        }

    except Exception as e:
        return {
            "file": getattr(filing, "accession_no", "unknown"),
            "status": f"error: {e}",
            "ticker": ticker,
        }


def _extract_text(filing_obj) -> str:
    """
    Extract human-readable text from an edgartools filing object.

    edgartools returns different object types depending on form type.
    We try multiple extraction methods in order of preference.
    """
    # Method 1: direct .text() method (TenK, TenQ objects)
    if hasattr(filing_obj, "text"):
        try:
            return filing_obj.text()
        except Exception:
            pass

    # Method 2: markdown representation
    if hasattr(filing_obj, "markdown"):
        try:
            return filing_obj.markdown()
        except Exception:
            pass

    # Method 3: string representation
    text = str(filing_obj)
    if len(text) > 5000:
        return text

    return ""


def _derive_quarter(filing_date: str, form: str) -> str | None:
    """Derive quarter label from filing date for 10-Q forms."""
    if form == "10-K":
        return None
    try:
        month = int(filing_date[5:7])
        year = filing_date[:4]
        q = (month - 1) // 3 + 1
        return f"{year}Q{q}"
    except (ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment corpus with additional EDGAR filings")
    parser.add_argument("--ticker", help="Single ticker to fetch (e.g. NVDA)")
    parser.add_argument("--all-corpus-tickers", action="store_true",
                        help=f"Fetch for all corpus tickers: {CORPUS_TICKERS}")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"],
                        help="Filing form type")
    parser.add_argument("--count", type=int, default=2, help="Number of recent filings per ticker")
    parser.add_argument("--corpus", default="data/edgar_corpus", help="Output corpus directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    args = parser.parse_args()

    if not args.ticker and not args.all_corpus_tickers:
        parser.error("Provide --ticker TICKER or --all-corpus-tickers")

    output_dir = Path(args.corpus)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = CORPUS_TICKERS if args.all_corpus_tickers else [args.ticker.upper()]

    console.print(f"\n[bold]EDGAR Corpus Augmentor[/bold]")
    console.print(f"Form: {args.form} | Count: {args.count} per ticker | Dry run: {args.dry_run}")
    console.print(f"Output: {output_dir}\n")

    all_results = []
    for ticker in tickers:
        with console.status(f"[dim]Fetching {args.form} filings for {ticker}…[/dim]"):
            results = fetch_filings(ticker, args.form, args.count, output_dir, args.dry_run)
        all_results.extend(results)

    # Results table
    table = Table("Ticker", "File", "Status", "Size", box=box.SIMPLE_HEAVY, header_style="bold yellow")
    ok = error = skipped = 0
    for r in all_results:
        size = f"{r.get('chars', 0):,} chars" if r.get("chars") else "—"
        status = r["status"]
        style = "green" if status == "ok" else ("yellow" if "skipped" in status else "red")
        table.add_row(r["ticker"], r["file"][:50], f"[{style}]{status}[/{style}]", size)
        if status == "ok":
            ok += 1
        elif "error" in status:
            error += 1
        else:
            skipped += 1

    console.print(table)
    console.print(f"\n[green]✓ {ok} fetched[/green]  [yellow]⊘ {skipped} skipped[/yellow]  [red]✗ {error} errors[/red]")

    if ok > 0 and not args.dry_run:
        console.print(
            f"\n[dim]Re-run the index builder to incorporate new files:[/dim]\n"
            f"  uv run python scripts/build_index.py --force"
        )


if __name__ == "__main__":
    main()
