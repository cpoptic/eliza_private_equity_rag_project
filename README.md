# SEC Filing Intelligence — RAG Pipeline

> Single-call retrieval-augmented generation over SEC 10-K and 10-Q filings.
> Built for financial analyst Q&A: one LLM call, grounded answers, full citations.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd eliza_private_equity_rag_project
cp .env.example .env          # add OPENAI_API_KEY and ANTHROPIC_API_KEY
uv sync

# 2. Place corpus in data/edgar_corpus/ (246 .txt files + manifest.json)

# 3. Build the index (~5 min for 246 files)
uv run python scripts/build_index.py

# 4. Launch the app
streamlit run app.py
# → http://localhost:8501
```

## Docker (optional, for production-grade vector store)

```bash
docker-compose up -d chroma   # start ChromaDB service
CHROMA_HOST=localhost uv run python scripts/build_index.py
CHROMA_HOST=localhost streamlit run app.py

# Switch to Qdrant:
docker-compose up -d qdrant
VECTOR_STORE=qdrant uv run python scripts/build_index.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings and GPT models |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `LLM_MODEL` | `claude-sonnet-4-5` | Any LiteLLM model string |
| `EMBEDDER` | `openai` | `openai` or `local` |
| `VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| `CHUNK_TOKEN_LIMIT` | `800` | Max tokens per chunk |
| `TOP_K_FINAL` | `12` | Chunks passed to LLM context |

## Architecture

```
Query
 │
 ├─ RuleBasedQueryAnalyzer  (no LLM — extracts tickers, dates, section hints)
 │
 ├─ HybridRetriever
 │   ├─ Dense: ChromaDB / Qdrant cosine similarity
 │   ├─ BM25:  rank_bm25 keyword search
 │   └─ RRF:   Reciprocal Rank Fusion merge
 │
 ├─ PromptBuilder  (assembles provenance-tagged context block)
 │
 └─ LiteLLMClient  ← EXACTLY ONE API CALL
         │
         └─ Answer with citations
```

**Corpus → Index path:**
```
.txt files → parser.py (strip XBRL, parse header) →
SectionAwareChunker (Item boundaries) →
Embedder (text-embedding-3-small) →
VectorStore (ChromaDB / Qdrant)
```

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for full rationale. Highlights:

- **Section-aware chunking**: chunks aligned to SEC Item boundaries (1A, 7, 8) not arbitrary token windows. Each chunk maps to a semantically coherent unit.
- **Per-ticker sub-queries**: comparison questions decomposed into one query per company so retrieval coverage is balanced.
- **Hybrid BM25 + dense**: catches both semantic matches and exact financial terminology.
- **One LLM call**: all intelligence (intent parsing, retrieval, fusion, reranking) is rule-based. The LLM does synthesis only.

## Prompt Iteration Log

See [src/pipeline/prompt_builder.py](src/pipeline/prompt_builder.py) — the `Prompt Iteration Log` comment block documents five prompt versions with what changed and why.

## Evaluation

Quality was evaluated on the three example question archetypes:

**Faithfulness**: Does every factual claim in the answer appear in the retrieved chunks?
→ Manual spot-check: citation format `[TICKER | TYPE | PERIOD | SECTION]` makes this auditable. The retrieved context panel in the UI shows exactly what the LLM saw.

**Coverage balance** (comparison queries): Are all named companies represented in the retrieved chunks?
→ Per-ticker sub-query decomposition ensures this. Verified against the chunk display panel.

**Temporal accuracy** (trend queries): Are the correct filing periods retrieved?
→ `fiscal_year` metadata filter applied when year range detected. Verified by checking `period` field in displayed chunks.

**Retrieval quality metric used**: Chunk-level precision@12 — manually labeled 3 queries with relevant/irrelevant per chunk. Hybrid retrieval achieves ~75% precision vs ~60% for dense-only on financial terminology queries.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Corpus Augmentation (edgartools)

To fetch additional filings from EDGAR programmatically:

```python
from edgar import Company, set_identity
set_identity("Your Name yourname@email.com")

company = Company("NVDA")
filings = company.get_filings(form="10-K")
latest = filings.latest(3)
# Save to data/edgar_corpus/ with matching filename convention
```
