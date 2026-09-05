import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_text, chunk_corpus  # noqa: E402


def test_chunk_text_respects_header_boundaries():
    text = "# Title\n\n## Section A\nSentence one. Sentence two.\n\n## Section B\nOther sentence."
    chunks = chunk_text(text, doc_id="doc1", chunk_size=100, overlap=10)
    sections = {c.section for c in chunks}
    assert "Section A" in sections
    assert "Section B" in sections
    # a chunk from Section A should not contain Section B's sentence
    for c in chunks:
        if c.section == "Section A":
            assert "Other sentence" not in c.text


def test_chunk_text_no_empty_chunks():
    text = "# Title\n\n## Section A\nHello world. This is a test.\n"
    chunks = chunk_text(text, doc_id="doc1", chunk_size=5, overlap=1)
    assert all(c.text.strip() for c in chunks)


def test_chunk_text_small_chunk_size_produces_multiple_chunks():
    text = "# T\n\n## S\n" + " ".join(f"Sentence number {i}." for i in range(30))
    chunks = chunk_text(text, doc_id="doc1", chunk_size=10, overlap=2)
    assert len(chunks) > 1


def test_chunk_ids_are_unique():
    text = "# T\n\n## S\n" + " ".join(f"Sentence number {i}." for i in range(30))
    chunks = chunk_text(text, doc_id="doc1", chunk_size=10, overlap=2)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_corpus_handles_multiple_docs():
    docs = {
        "a": "# A\n\n## Sec\nHello there. General Kenobi.",
        "b": "# B\n\n## Sec\nAnother document. With more text.",
    }
    chunks = chunk_corpus(docs, chunk_size=100, overlap=10)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"a", "b"}


def test_empty_document_produces_no_chunks():
    chunks = chunk_text("# Title\n\n## Empty\n", doc_id="doc1", chunk_size=100, overlap=10)
    assert chunks == []
