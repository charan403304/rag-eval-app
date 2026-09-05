"""
Metrics for the eval harness.

Deliberately mixes three kinds of signal, because any one alone is
misleading:

  - Retrieval metrics (precision/recall against labeled relevant docs) tell
    you if the retriever found the right material, independent of whether
    the LLM used it well.
  - Citation validity tells you if the generated answer's [n] markers point
    at real, retrieved chunks -- a cheap, deterministic hallucination check
    that doesn't need an LLM judge.
  - Keyword recall + optional LLM-as-judge tell you if the answer content
    was actually correct, which retrieval metrics alone can't capture (you
    can retrieve the right doc and still generate a wrong answer from it).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.pipeline import RAGResponse


@dataclass
class RetrievalMetrics:
    precision: float
    recall: float
    hit_at_k: bool  # was at least one relevant doc retrieved at all


def retrieval_precision_recall(response: RAGResponse, relevant_doc_ids: list[str]) -> RetrievalMetrics:
    if not relevant_doc_ids:
        # unanswerable question: "correct" retrieval is retrieving nothing
        # relevant, so precision/recall are undefined and excluded upstream.
        return RetrievalMetrics(precision=0.0, recall=0.0, hit_at_k=False)

    retrieved_doc_ids = [rc.chunk.doc_id for rc in response.retrieved]
    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    hit = true_positives > 0
    return RetrievalMetrics(precision=precision, recall=recall, hit_at_k=hit)


def citation_validity(response: RAGResponse) -> float:
    """Fraction of citation markers in the answer that point at a real,
    retrieved chunk index. 1.0 if no citations were expected/made on a
    refusal. This is a hallucination proxy: a model citing [5] when only
    4 chunks were provided, or citing nothing while making factual claims,
    both hurt this score."""
    if response.refused:
        return 1.0
    n_chunks = len(response.retrieved)
    if n_chunks == 0:
        return 0.0
    if not response.citations_used:
        return 0.0  # made claims with no citation at all -> worst case
    valid = sum(1 for c in response.citations_used if 1 <= c <= n_chunks)
    return valid / len(response.citations_used)


def keyword_recall(response: RAGResponse, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    text = response.answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)


def refusal_correct(response: RAGResponse, answerable: bool) -> bool:
    """True if the system refused exactly when it should have, and
    answered exactly when it should have."""
    return response.refused != answerable


# --- Optional LLM-as-judge -------------------------------------------------

JUDGE_PROMPT = """You are grading a RAG system's answer for factual correctness \
against a reference. Respond with ONLY a single integer from 1 to 5:

5 = fully correct and complete
4 = correct but missing minor detail
3 = partially correct
2 = mostly wrong but on-topic
1 = wrong or off-topic

Question: {query}
Expected key facts: {expected_keywords}
System's answer: {answer}

Respond with only the digit.
"""


def llm_judge_score(query: str, answer: str, expected_keywords: list[str]) -> int | None:
    """Requires ANTHROPIC_API_KEY. Returns None (and the caller should fall
    back to keyword_recall) if no key is configured -- this keeps `make
    eval` runnable offline, with the LLM judge as an opt-in upgrade for a
    more nuanced correctness signal than keyword matching alone."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()
    prompt = JUDGE_PROMPT.format(
        query=query, expected_keywords=", ".join(expected_keywords), answer=answer
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    try:
        return int(text[0])
    except (ValueError, IndexError):
        return None
