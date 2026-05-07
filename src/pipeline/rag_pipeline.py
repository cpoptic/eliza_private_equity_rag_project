"""
RAGPipeline: orchestrates the full parse → chunk → embed → index → query flow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseLLMClient,
    BaseQueryAnalyzer,
    BaseRetriever,
    BaseVectorStore,
    QueryContext,
    RetrievedChunk,
)
from src.pipeline.prompt_builder import PromptBuilder
from src.preprocessing.parser import FilingParser

logger = logging.getLogger(__name__)

_EMBED_BATCH = 100   # chunks per embedding call


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
        progress_callback=None,
    ) -> dict:
        corpus_path = Path(corpus_dir)
        txt_files = sorted(corpus_path.glob("*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in {corpus_dir}")

        if force:
            logger.info("Force re-index: dropping existing collection")
            self._store.delete_collection()

        # Build set of already-indexed chunk IDs for incremental updates
        existing_ids: set[str] = set()
        if not force and self._store.collection_exists():
            existing_ids = {c.chunk_id for c in self._store.get_all_chunks()}
            logger.info("Incremental mode: %d chunks already indexed", len(existing_ids))

        total_files = len(txt_files)
        total_chunks_added = 0
        files_processed = 0
        files_skipped = 0

        for i, filepath in enumerate(txt_files):
            try:
                metadata, text = self._parser.parse(filepath)
                chunks = self._chunker.chunk(text, metadata)

                new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
                if not new_chunks:
                    files_skipped += 1
                else:
                    for j in range(0, len(new_chunks), _EMBED_BATCH):
                        batch = new_chunks[j : j + _EMBED_BATCH]
                        texts = [c.text for c in batch]
                        embeddings = self._embedder.embed_texts(texts)
                        self._store.add_chunks(batch, embeddings)
                        total_chunks_added += len(batch)

                    files_processed += 1

            except Exception as exc:
                logger.warning("Skipping %s: %s", filepath.name, exc)

            if progress_callback:
                progress_callback(i + 1, total_files, filepath.name)

        return {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "chunks_added": total_chunks_added,
            "total_indexed": self._store.count(),
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 12) -> QueryResult:
        t0 = time.perf_counter()

        context = self._query_analyzer.analyze(question)
        chunks = self._retriever.retrieve(context, top_k=top_k)

        system = self._prompt_builder.build_system_prompt()
        prompt = self._prompt_builder.build_user_prompt(question, chunks, context.query_type)
        answer = self._llm.complete(prompt, system=system)

        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResult(
            answer=answer,
            chunks=chunks,
            query_context=context,
            latency_ms=latency_ms,
        )

    def is_indexed(self) -> bool:
        return self._store.collection_exists() and self._store.count() > 0
