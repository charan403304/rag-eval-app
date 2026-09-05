"""The RAGPipeline class -- the single entry point the Streamlit app,
eval harness, and any future API layer all call into. Keeping one shared
entry point is what makes the evaluation harness trustworthy: eval is
running the exact same code path as the live app, not a parallel
reimplementation that could silently drift from it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import generator
from .bm25 import BM25
from .chunking import Chunk, chunk_corpus
from .config import CONFIG, CORPUS_DIR, INDEX_DIR
from .cost_tracker import QueryStats
from .embeddings import Embedder, get_embedder
from .retriever import HybridRetriever, RetrievedChunk
from .vectorstore import VectorStore


@dataclass
class RAGResponse:
    query: str
    answer: str
    refused: bool
    retrieved: list[RetrievedChunk]
    citations_used: list[int]
    stats: QueryStats


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, str]:
    docs = {}
    for path in sorted(corpus_dir.glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def build_index(corpus_dir: Path = CORPUS_DIR, embedder: Embedder | None = None
                 ) -> tuple[VectorStore, BM25, list[Chunk]]:
    embedder = embedder or get_embedder(CONFIG.embedding_provider)
    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, CONFIG.chunk_size_tokens, CONFIG.chunk_overlap_tokens)

    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore()
    store.add(vectors, chunks)

    bm25 = BM25()
    bm25.fit([c.chunk_id for c in chunks], [c.text for c in chunks])

    return store, bm25, chunks


class RAGPipeline:
    def __init__(self, store: VectorStore, bm25: BM25, embedder: Embedder | None = None,
                 llm_provider: str | None = None):
        self.embedder = embedder or get_embedder(CONFIG.embedding_provider)
        self.retriever = HybridRetriever(store, bm25, self.embedder)
        self.backend = generator.get_backend(llm_provider or CONFIG.llm_provider)

    @classmethod
    def from_corpus(cls, corpus_dir: Path = CORPUS_DIR) -> "RAGPipeline":
        embedder = get_embedder(CONFIG.embedding_provider)
        store, bm25, _ = build_index(corpus_dir, embedder)
        return cls(store, bm25, embedder)

    @classmethod
    def from_saved_index(cls, index_dir: Path = INDEX_DIR) -> "RAGPipeline":
        embedder = get_embedder(CONFIG.embedding_provider)
        store = VectorStore.load(index_dir)
        docs = load_corpus(CORPUS_DIR)
        bm25 = BM25()
        bm25.fit([c.chunk_id for c in store.chunks], [c.text for c in store.chunks])
        return cls(store, bm25, embedder)

    def answer(self, query: str, top_k: int | None = None) -> RAGResponse:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not retrieved or retrieved[0].score < CONFIG.min_relevance_score:
            stats = QueryStats(provider="none", model="none", retrieval_ms=retrieval_ms,
                                total_ms=retrieval_ms)
            return RAGResponse(
                query=query,
                answer="I don't have enough information in the documentation to answer that.",
                refused=True,
                retrieved=retrieved,
                citations_used=[],
                stats=stats,
            )

        result = generator.generate_answer(query, retrieved, self.backend)
        result.stats.retrieval_ms = retrieval_ms
        result.stats.total_ms = retrieval_ms + result.stats.generation_ms

        return RAGResponse(
            query=query,
            answer=result.text,
            refused=result.refused,
            retrieved=retrieved,
            citations_used=result.citations_used,
            stats=result.stats,
        )
