"""
Minimal BM25 (Okapi) implementation in pure Python + numpy.

Implemented by hand rather than pulling in `rank_bm25` for two reasons:
it's ~40 lines, and it means the whole repo has zero hard runtime
dependencies beyond numpy for the offline path (see embeddings.py for the
same reasoning). BM25 catches exact keyword/identifier matches ("v1",
"429", "healthz") that a semantic embedding can blur together -- that's
the whole reason we hybridize instead of using vectors alone.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "for", "with", "as", "at", "by", "this", "that",
    "it", "its", "we", "our", "not", "do", "does", "if", "than", "then",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.n_docs: int = 0

    def fit(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.n_docs = len(texts)
        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            self.doc_lens.append(len(tokens))
            for term in counts:
                self.df[term] += 1
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.n_docs)

    def _idf(self, term: str) -> float:
        n_qualify = self.df.get(term, 0)
        # BM25+ style idf, floored at a small positive value so common-ish
        # terms don't get a negative weight
        return math.log(1 + (self.n_docs - n_qualify + 0.5) / (n_qualify + 0.5))

    def score(self, query: str) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = [0.0] * self.n_docs
        for term in q_tokens:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i in range(self.n_docs):
                f = self.doc_freqs[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avg_doc_len))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked
