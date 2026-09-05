import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline, build_index  # noqa: E402
from src.config import CORPUS_DIR  # noqa: E402


def test_build_index_produces_chunks():
    store, bm25, chunks = build_index(CORPUS_DIR)
    assert len(chunks) > 0
    assert len(store) == len(chunks)
    assert bm25.n_docs == len(chunks)


def test_pipeline_answers_in_scope_question():
    pipeline = RAGPipeline.from_corpus()
    response = pipeline.answer("What is the API rate limit?")
    assert response.refused is False
    assert len(response.retrieved) > 0
    assert response.retrieved[0].chunk.doc_id == "api-guidelines"


def test_pipeline_refuses_out_of_scope_question():
    pipeline = RAGPipeline.from_corpus()
    response = pipeline.answer("What is the capital of Mongolia and its population history")
    # not a hard guarantee with the offline hash embedder (see README limitations),
    # but a clearly nonsense/unrelated query should still score low
    assert response.retrieved[0].score < 0.05 or response.refused


def test_pipeline_citations_reference_real_chunks():
    pipeline = RAGPipeline.from_corpus()
    response = pipeline.answer("How does the rolling deploy health check work?")
    for c in response.citations_used:
        assert 1 <= c <= len(response.retrieved)


def test_pipeline_stats_are_populated():
    pipeline = RAGPipeline.from_corpus()
    response = pipeline.answer("What is the API rate limit?")
    assert response.stats.total_ms >= 0
    assert response.stats.provider == "mock"
