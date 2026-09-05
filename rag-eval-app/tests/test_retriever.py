import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25, tokenize  # noqa: E402
from src.chunking import Chunk  # noqa: E402
from src.embeddings import HashEmbedder  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402


def test_tokenize_strips_stopwords():
    tokens = tokenize("The quick brown fox is in the box")
    assert "the" not in tokens
    assert "quick" in tokens


def test_bm25_ranks_exact_match_highest():
    bm25 = BM25()
    docs = ["cats are great pets", "dogs are loyal animals", "the weather today is sunny"]
    bm25.fit(["d1", "d2", "d3"], docs)
    ranked = bm25.score("loyal dogs")
    assert ranked[0][0] == "d2"


def test_bm25_zero_score_for_no_overlap():
    bm25 = BM25()
    bm25.fit(["d1"], ["completely unrelated content here"])
    ranked = bm25.score("xyzzyxyzzy")
    assert ranked[0][1] == 0.0


def _make_chunk(doc_id: str, chunk_id: str, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_id=chunk_id, text=text, section="s", start_char=0, end_char=len(text))


def test_hybrid_retriever_returns_top_k():
    texts = [
        "Deployments use a rolling strategy with health checks.",
        "The API rate limit is 100 requests per minute.",
        "Incidents are classified as SEV1, SEV2, or SEV3.",
        "Database migrations must be reversible.",
    ]
    chunks = [_make_chunk("doc", f"doc::{i}", t) for i, t in enumerate(texts)]
    embedder = HashEmbedder(dim=64)
    vectors = embedder.embed(texts)

    store = VectorStore()
    store.add(vectors, chunks)

    bm25 = BM25()
    bm25.fit([c.chunk_id for c in chunks], texts)

    retriever = HybridRetriever(store, bm25, embedder)
    results = retriever.retrieve("what is the API rate limit", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc::1"


def test_hybrid_retriever_scores_are_sorted_descending():
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    chunks = [_make_chunk("doc", f"doc::{i}", t) for i, t in enumerate(texts)]
    embedder = HashEmbedder(dim=32)
    vectors = embedder.embed(texts)
    store = VectorStore()
    store.add(vectors, chunks)
    bm25 = BM25()
    bm25.fit([c.chunk_id for c in chunks], texts)
    retriever = HybridRetriever(store, bm25, embedder)
    results = retriever.retrieve("alpha beta", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
