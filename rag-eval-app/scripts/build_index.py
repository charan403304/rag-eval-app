#!/usr/bin/env python3
"""Build the vector index from data/corpus and persist it to data/index/.

Run this once after changing the corpus or the embedding provider:

    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, INDEX_DIR  # noqa: E402
from src.pipeline import build_index  # noqa: E402


def main():
    print(f"Building index with embedding provider: {CONFIG.embedding_provider}")
    store, bm25, chunks = build_index()
    print(f"Indexed {len(chunks)} chunks from the corpus.")
    store.save(INDEX_DIR)
    print(f"Saved vector index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
