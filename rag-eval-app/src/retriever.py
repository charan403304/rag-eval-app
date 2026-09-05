"""
Hybrid retrieval: BM25 (sparse, keyword) + vector search (dense, semantic),
combined with Reciprocal Rank Fusion (RRF).

Why RRF instead of averaging raw scores: BM25 scores and cosine
similarities live on completely different, uncalibrated scales, so a
weighted sum of the raw numbers is meaningless without hand-tuned
normalization per corpus. RRF sidesteps that: it only looks at each
document's *rank* in each list, not the raw score, which makes it robust
out of the box and is the standard trick for this exact problem
(originally from Cormack et al., 2009).
"""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunking import Chunk
from .config import CONFIG
from .embeddings import Embedder
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # fused RRF score
    dense_rank: int | None
    sparse_rank: int | None


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25, embedder: Embedder):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self._chunk_by_id = {c.chunk_id: c for c in vector_store.chunks}

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or CONFIG.top_k_final

        query_vec = self.embedder.embed([query])[0]
        dense_hits = self.vector_store.search(query_vec, top_k=CONFIG.top_k_dense)
        dense_rank = {c.chunk_id: i for i, (c, _score) in enumerate(dense_hits)}

        sparse_hits = self.bm25.score(query)[: CONFIG.top_k_sparse]
        sparse_rank = {chunk_id: i for i, (chunk_id, _score) in enumerate(sparse_hits)
                        if _score > 0}

        all_ids = set(dense_rank) | set(sparse_rank)
        fused: list[RetrievedChunk] = []
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (CONFIG.rrf_k + dense_rank[chunk_id] + 1)
            if chunk_id in sparse_rank:
                score += 1.0 / (CONFIG.rrf_k + sparse_rank[chunk_id] + 1)
            fused.append(RetrievedChunk(
                chunk=self._chunk_by_id[chunk_id],
                score=score,
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
            ))

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]
