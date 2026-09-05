"""
A deliberately simple vector store: numpy matrix + cosine similarity via
matrix multiply, persisted as .npy + .json. For this corpus size (dozens to
low thousands of chunks) brute-force cosine search is sub-millisecond and
outperforms the operational cost of running a vector DB service. The
`VectorStore` interface is narrow on purpose (`add`, `search`, `save`,
`load`) so swapping in Chroma/pgvector/Pinecone later is a one-file change,
not a rewrite -- see the README's "scaling this up" section.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None  # (n, dim)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("vectors and chunks must be the same length")
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        # vectors are pre-normalized, so dot product == cosine similarity
        sims = self.vectors @ query_vec
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = [asdict(c) for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        store = cls()
        store.vectors = np.load(directory / "vectors.npy")
        meta = json.loads((directory / "chunks.json").read_text())
        store.chunks = [Chunk(**m) for m in meta]
        return store

    def __len__(self) -> int:
        return len(self.chunks)
