"""
Chunking strategy.

Design decision (documented here on purpose -- this is the kind of thing an
interviewer will ask about): we chunk by a *target token count with overlap*,
but we snap chunk boundaries to the nearest sentence end within a small
window instead of cutting mid-sentence. Pure fixed-size chunking is simpler
but regularly slices a sentence in half, which measurably hurts retrieval
quality because the resulting chunk embedding represents a sentence
fragment rather than a coherent idea. Header-aware splitting (never crossing
a markdown `##` boundary) is applied first, so a chunk never blends two
unrelated sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADER_LINE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section: str  # nearest markdown header above this chunk, for citation display
    start_char: int
    end_char: int


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (header, body) sections on '##' headers."""
    matches = list(_HEADER_LINE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    # preamble before the first header (usually just the H1 title)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((matches[0].group().strip("# ").strip(), preamble))

    for i, m in enumerate(matches):
        header = m.group().strip("# ").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((header, body))
    return sections


def _word_count(s: str) -> int:
    return len(s.split())


def chunk_text(text: str, doc_id: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a single markdown document into overlapping, sentence-safe chunks."""
    sections = _split_sections(text)
    chunks: list[Chunk] = []
    cursor = 0  # approximate global char offset, for citation/debugging purposes
    chunk_idx = 0

    for header, body in sections:
        sentences = [s.strip() for s in _SENTENCE_END.split(body) if s.strip()]
        if not sentences:
            continue

        i = 0
        while i < len(sentences):
            buf: list[str] = []
            count = 0
            j = i
            while j < len(sentences) and count < chunk_size:
                buf.append(sentences[j])
                count += _word_count(sentences[j])
                j += 1

            chunk_str = " ".join(buf)
            start_char = cursor
            end_char = cursor + len(chunk_str)
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::{chunk_idx}",
                text=chunk_str,
                section=header or doc_id,
                start_char=start_char,
                end_char=end_char,
            ))
            chunk_idx += 1
            cursor = end_char + 1

            if j >= len(sentences):
                break

            # step back by ~overlap words worth of sentences so consecutive
            # chunks share context (helps when the answer straddles a
            # sentence-window boundary)
            back = 0
            k = j - 1
            while k > i and back < overlap:
                back += _word_count(sentences[k])
                k -= 1
            i = max(k + 1, i + 1)  # guarantee forward progress

    return chunks


def chunk_corpus(docs: dict[str, str], chunk_size: int, overlap: int) -> list[Chunk]:
    """docs: {doc_id: raw_text}. Returns all chunks across the corpus."""
    all_chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(text, doc_id, chunk_size, overlap))
    return all_chunks
