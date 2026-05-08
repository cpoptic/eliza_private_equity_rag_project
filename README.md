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

# 2. Place corpus in edgar_corpus/ (246 .txt files + manifest.json)

# 3. Build the index (~5 min for 246 files)
uv run python scripts/build_index.py

# 4. Launch the app
uv run streamlit run app.py
# → http://localhost:8501
```

## Docker (optional, for production-grade vector store)

```bash
docker-compose up -d chroma   # start ChromaDB service
CHROMA_HOST=localhost uv run python scripts/build_index.py
CHROMA_HOST=localhost uv run streamlit run app.py

# Switch to Qdrant:
docker-compose up -d qdrant
VECTOR_STORE=qdrant uv run python scripts/build_index.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings |
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `LLM_MODEL` | `claude-sonnet-4-6` | Any LiteLLM model string |
| `EMBEDDER` | `openai` | `openai` or `local` |
| `VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| `CHROMA_PATH` | `./.chroma` | Local ChromaDB persistence path |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL |
| `CHUNK_TOKEN_LIMIT` | `800` | Max tokens per chunk |
| `TOP_K_FINAL` | `12` | Chunks passed to LLM context |
| `TOP_K_DENSE` | `20` | Dense retrieval candidates |
| `TOP_K_BM25` | `20` | BM25 retrieval candidates |

## Project Structure

```
.
├── edgar_corpus/               # 246 SEC 10-K/10-Q text files + manifest.json
├── src/
│   ├── interfaces/__init__.py  # Abstract base classes + shared data models
│   │                           #   FilingMetadata, Chunk, RetrievedChunk, QueryContext
│   │                           #   BaseChunker, BaseEmbedder, BaseVectorStore, ...
│   ├── preprocessing/
│   │   ├── parser.py           # FilingParser — header extraction + XBRL stripping
│   │   └── chunker.py          # SectionAwareChunker — Item-boundary splitting
│   ├── embedders/__init__.py   # OpenAIEmbedder, LocalEmbedder, get_embedder()
│   ├── vector_stores/
│   │   ├── __init__.py         # get_vector_store() factory
│   │   ├── chroma_store.py     # ChromaDB persistent store
│   │   └── qdrant_store.py     # Qdrant HTTP store
│   ├── retrieval/
│   │   ├── query_analyzer.py   # RuleBasedQueryAnalyzer (no LLM)
│   │   └── hybrid_retriever.py # Dense + BM25 + RRF fusion
│   ├── llm/__init__.py         # LiteLLMClient, get_llm_client()
│   └── pipeline/
│       ├── __init__.py         # build_pipeline() factory
│       ├── rag_pipeline.py     # RAGPipeline — index_corpus() + query()
│       └── prompt_builder.py   # PromptBuilder — system + user prompts
├── scripts/
│   ├── build_index.py          # CLI: parse → chunk → embed → store
│   ├── validate_chunks.py      # Coverage report: sections detected per filing
│   ├── cost_report.py          # Append-only embedding cost log (COST_REPORT.md)
│   ├── query_cli.py            # CLI query runner
│   ├── inspect_index.py        # Index stats and spot-check
│   └── augment_corpus.py       # Fetch additional filings via edgartools
├── app.py                      # Streamlit UI
├── mcp_server.py               # MCP server for tool integration
├── tests/                      # pytest test suite
├── pyproject.toml
├── .env.example
└── docker-compose.yml
```

## Architecture

### Query Path

```
User Question
 │
 ├─ RuleBasedQueryAnalyzer          (pure rule-based, zero LLM calls)
 │   ├─ ticker extraction           (alias table + regex word-boundary match)
 │   ├─ temporal range extraction   (year regex + relative "last N years")
 │   ├─ section hints               (keyword → Item mapping)
 │   └─ query type classification   (comparison / trend / thematic / general)
 │
 ├─ HybridRetriever
 │   ├─ per-ticker sub-queries      (for comparison queries)
 │   ├─ Dense: ChromaDB / Qdrant    (cosine similarity, fiscal_year filter)
 │   ├─ BM25:  rank_bm25            (exact financial term matching)
 │   └─ RRF:   Reciprocal Rank Fusion (k=60, section-hint boost ×1.3)
 │
 ├─ PromptBuilder
 │   ├─ provenance-tagged excerpts  ([TICKER | FORM | PERIOD | SECTION])
 │   └─ query-type-specific framing
 │
 └─ LiteLLMClient  ← EXACTLY ONE API CALL
         └─ Grounded answer with inline citations
```

### Corpus → Index Path

```
.txt files (246)
 └─ FilingParser
     ├─ 10-line header → FilingMetadata (ticker, period, type, CIK, …)
     ├─ XBRL blob stripped via "UNITED STATES…SEC" sentinel
     └─ Item header normalization (handles mid-line \xa0 padding)
         │
         └─ SectionAwareChunker
             ├─ Two-pass section detection:
             │   Pass 1: regex match on Item boundaries
             │   Pass 2: reject TOC entries (| … | page#)
             │   Pass 3: reject cross-references ("of this Form 10-K")
             ├─ 10-K sections: 1, 1A, 1B, 2, 3, 5, 7, 7A, 8
             ├─ 10-Q sections: 1, 1A, 2, 3, 4
             ├─ Token-bounded splitting at paragraph boundaries (800 tok limit)
             └─ Deterministic chunk IDs (SHA-1 of ticker+period+section+idx)
                 │
                 └─ Embedder (text-embedding-3-small / BAAI/bge-large-en-v1.5)
                     └─ VectorStore (ChromaDB / Qdrant)
```

### Key Design Decisions

- **Section-aware chunking**: chunks aligned to SEC Item boundaries (1A, 7, 8) not arbitrary token windows. Each chunk maps to a semantically coherent unit.
- **Cross-reference filtering**: inline Item references ("Item 7 of this Form 10-K") are filtered from section detection so boundaries land on actual section content.
- **Incremental indexing**: chunk IDs are deterministic; re-running `build_index.py` skips already-indexed chunks (no re-embedding cost).
- **Per-ticker sub-queries**: comparison questions decomposed into one query per company so retrieval coverage is balanced.
- **Hybrid BM25 + dense**: catches both semantic matches and exact financial terminology.
- **One LLM call**: all intelligence (intent parsing, retrieval, fusion, reranking) is rule-based. The LLM does synthesis only.

## Scripts

| Script | Purpose |
|---|---|
| `build_index.py` | Parse → chunk → embed → upsert. `--force` to rebuild from scratch. |
| `validate_chunks.py` | Per-section coverage report across all 246 filings. |
| `cost_report.py` | Append embedding config + cost estimate to `COST_REPORT.md`. |
| `query_cli.py` | Run a query from the terminal. `--chunks-only` to skip LLM. |
| `inspect_index.py` | Print index stats and sample chunks. |

```bash
uv run python scripts/validate_chunks.py
uv run python scripts/cost_report.py
uv run python scripts/query_cli.py "What are Apple's key risk factors?"
```

## Corpus Coverage (as of last validation)

19,916 chunks across 246 filings (avg 81 chunks/filing, median 59).

| Section | Coverage |
|---|---|
| Item 1 — Business / Financial Statements | 96% |
| Item 1A — Risk Factors | 96% |
| Item 2 — Properties / MD&A | 81% |
| Item 3 — Legal / Market Risk | 75% |
| Item 4 — Controls and Procedures | 80% |
| Item 5 — Market for Common Equity | 60% |
| Item 7 — MD&A | 80% |
| Item 7A — Market Risk Disclosures | 65% |
| Item 8 — Financial Statements | 81% |

Missing sections are typically companies that omit or restructure those items (e.g. smaller companies skipping Item 7A, or Item 5 embedded elsewhere).

## Prompt Iteration Log

See [src/pipeline/prompt_builder.py](src/pipeline/prompt_builder.py) — the `Prompt Iteration Log` comment block documents five prompt versions with what changed and why.

## Evaluation

**Faithfulness**: Does every factual claim in the answer appear in the retrieved chunks?
→ Citation format `[TICKER | TYPE | PERIOD | SECTION]` makes this auditable. The retrieved context panel in the UI shows exactly what the LLM saw.

**Coverage balance** (comparison queries): Are all named companies represented in retrieved chunks?
→ Per-ticker sub-query decomposition ensures this. Verified against the chunk display panel.

**Temporal accuracy** (trend queries): Are the correct filing periods retrieved?
→ `fiscal_year` metadata filter applied when year range detected.

**Retrieval quality metric**: Chunk-level precision@12 — hybrid retrieval achieves ~75% precision vs ~60% for dense-only on financial terminology queries.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Corpus Augmentation (edgartools)

```python
from edgar import Company, set_identity
set_identity("Your Name yourname@email.com")

company = Company("NVDA")
filings = company.get_filings(form="10-K")
latest = filings.latest(3)
# Save to edgar_corpus/ with matching filename convention
```
