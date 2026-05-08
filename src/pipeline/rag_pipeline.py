"""
RAGPipeline: orchestrates the full parse → chunk → embed → index → query flow.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMClient,
    BaseQueryAnalyzer,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    QueryContext,
    RetrievedChunk,
)
from src.pipeline.prompt_builder import PromptBuilder
from src.preprocessing.parser import FilingParser

logger = logging.getLogger(__name__)

_EMBED_BATCH = 100   # chunks per embedding call
_PARSE_WORKERS = min(8, (os.cpu_count() or 4))   # parallel parse+chunk threads


@dataclass
class QueryResult:
    answer: str
    chunks: list[RetrievedChunk]
    query_context: QueryContext
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class RAGPipeline:

    def __init__(
        self,
        parser: FilingParser,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        retriever: BaseRetriever,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder,
        query_analyzer: BaseQueryAnalyzer,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = vector_store
        self._retriever = retriever
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._query_analyzer = query_analyzer

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_corpus(
        self,
        corpus_dir: str | Path,
        force: bool = False,
        parse_progress_callback=None,
        embed_progress_callback=None,
        # Legacy single callback — still accepted for back-compat
        progress_callback=None,
    ) -> dict:
        corpus_path = Path(corpus_dir)
        txt_files = sorted(corpus_path.glob("*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in {corpus_dir}")

        if force:
            logger.info("Force re-index: dropping existing collection")
            self._store.delete_collection()

        existing_ids: set[str] = set()
        if not force and self._store.collection_exists():
            existing_ids = {c.chunk_id for c in self._store.get_all_chunks()}
            logger.info("Incremental mode: %d chunks already indexed", len(existing_ids))

        total_files = len(txt_files)
        parse_errors: list[dict] = []
        all_new_chunks: list[Chunk] = []
        files_processed = 0
        files_skipped = 0

        # ── Phase 1: Parse + chunk (parallel) ────────────────────────────
        def _parse_chunk(filepath: Path) -> tuple[str, list[Chunk]]:
            metadata, text = self._parser.parse(filepath)
            chunks = self._chunker.chunk(text, metadata)
            new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            return filepath.name, new_chunks

        completed_files = 0
        with ThreadPoolExecutor(max_workers=_PARSE_WORKERS) as executor:
            future_map = {executor.submit(_parse_chunk, f): f for f in txt_files}
            for future in as_completed(future_map):
                filepath = future_map[future]
                completed_files += 1
                try:
                    name, new_chunks = future.result()
                    if new_chunks:
                        all_new_chunks.extend(new_chunks)
                        files_processed += 1
                    else:
                        files_skipped += 1
                    logger.debug("Parsed %s → %d new chunks", name, len(new_chunks))
                except Exception as exc:
                    logger.warning("Parse/chunk error %s: %s", filepath.name, exc)
                    parse_errors.append({"file": filepath.name, "error": str(exc)})

                # Fire legacy single callback (parse phase)
                if progress_callback:
                    progress_callback(completed_files, total_files, filepath.name)
                if parse_progress_callback:
                    parse_progress_callback(completed_files, total_files, filepath.name)

        # ── Phase 2: Embed + store (sequential, rate-limit-safe) ─────────
        total_to_embed = len(all_new_chunks)
        chunks_added = 0
        embed_errors: list[dict] = []

        has_with_ids = hasattr(self._embedder, "embed_texts_with_ids")
        total_batches = max(1, (total_to_embed + _EMBED_BATCH - 1) // _EMBED_BATCH)
        completed_batches = 0

        for j in range(0, total_to_embed, _EMBED_BATCH):
            batch = all_new_chunks[j : j + _EMBED_BATCH]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]

            try:
                if has_with_ids:
                    embeddings, failed_ids = self._embedder.embed_texts_with_ids(texts, ids)
                    # Filter out failed chunks
                    if failed_ids:
                        failed_set = set(failed_ids)
                        good = [(c, e) for c, e in zip(batch, embeddings)
                                if c.chunk_id not in failed_set]
                        # Log with section context
                        for c in batch:
                            if c.chunk_id in failed_set:
                                logger.error(
                                    "Dropped oversized chunk %s [section: %s, ~%d tokens]",
                                    c.chunk_id, c.section, c.token_count,
                                )
                                embed_errors.append({
                                    "chunk_id": c.chunk_id,
                                    "section": c.section,
                                    "token_count": c.token_count,
                                })
                        if good:
                            good_chunks, good_embs = zip(*good)
                            self._store.add_chunks(list(good_chunks), list(good_embs))
                            chunks_added += len(good_chunks)
                    else:
                        self._store.add_chunks(batch, embeddings)
                        chunks_added += len(batch)
                else:
                    embeddings = self._embedder.embed_texts(texts)
                    self._store.add_chunks(batch, embeddings)
                    chunks_added += len(batch)

            except Exception as exc:
                logger.warning(
                    "Embedding batch [%d:%d] failed: %s — skipping %d chunks",
                    j, j + len(batch), exc, len(batch),
                )
                embed_errors.append({"batch_start": j, "error": str(exc)})

            completed_batches += 1
            if embed_progress_callback:
                embed_progress_callback(completed_batches, total_batches, len(batch))

        return {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "chunks_added": chunks_added,
            "total_indexed": self._store.count(),
            "parse_errors": parse_errors,
            "embed_errors": embed_errors,
            "total_to_embed": total_to_embed,
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 12, profile: bool = False) -> QueryResult:
        t0 = time.perf_counter()
        timing: dict[str, float] = {}

        def _elapsed_ms() -> float:
            return (time.perf_counter() - t0) * 1000

        ts = time.perf_counter()
        context = self._query_analyzer.analyze(question)
        if profile:
            timing["query_analysis_ms"] = (time.perf_counter() - ts) * 1000

        ts = time.perf_counter()
        chunks = self._retriever.retrieve(context, top_k=top_k)
        if profile:
            timing["retrieval_ms"] = (time.perf_counter() - ts) * 1000

        ts = time.perf_counter()
        system = self._prompt_builder.build_system_prompt()
        prompt = self._prompt_builder.build_user_prompt(question, chunks, context.query_type)
        if profile:
            timing["prompt_build_ms"] = (time.perf_counter() - ts) * 1000

        ts = time.perf_counter()
        answer = self._llm.complete(prompt, system=system)
        if profile:
            timing["llm_complete_ms"] = (time.perf_counter() - ts) * 1000

        latency_ms = _elapsed_ms()
        return QueryResult(
            answer=answer,
            chunks=chunks,
            query_context=context,
            latency_ms=latency_ms,
            metadata={"timing": timing} if profile else {},
        )

    def stream_query(self, question: str, top_k: int = 12):
        """
        Streaming variant of query().

        Runs analysis + retrieval synchronously, then yields answer tokens
        from the LLM as they arrive. Returns a tuple:
            (context, chunks, token_generator, retrieval_latency_ms)

        The caller is responsible for collecting the full answer if it needs
        to save or display it after streaming completes.
        """
        t0 = time.perf_counter()
        context = self._query_analyzer.analyze(question)
        chunks = self._retriever.retrieve(context, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        system = self._prompt_builder.build_system_prompt()
        prompt = self._prompt_builder.build_user_prompt(question, chunks, context.query_type)

        token_stream = self._llm.stream_complete(prompt, system=system)
        return context, chunks, token_stream, retrieval_ms

    def is_indexed(self) -> bool:
        return self._store.collection_exists() and self._store.count() > 0
