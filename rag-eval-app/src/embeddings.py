"""
Pluggable embedding backends.

Why a hash-based offline embedder exists at all: this repo's tests, CI, and
`eval/run_eval.py` need to run without secrets and without network access
(e.g. in a CI runner or a reviewer's machine with no API key configured).
A hashing-trick embedder (random projection of char n-gram hashes into a
fixed-dim space, L2-normalized) gives a real, deterministic, non-trivial
vector space -- semantically weaker than a learned embedding model, but
good enough to prove the retrieval/fusion/eval machinery actually works.
Swapping `EMBEDDING_PROVIDER=voyage` or `openai` in .env drops in a real
model with no other code changes, because everything downstream only
depends on the `Embedder.embed(texts) -> np.ndarray` interface.
"""
from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic offline embedder: char 3-5-gram hashing trick + L2 norm.

    Not a substitute for a learned model -- it has no real semantic
    generalization (it can't tell "car" and "automobile" are related). It's
    here so the pipeline is fully exercised offline; see module docstring.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> list[str]:
        text = text.lower()
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
        # also include whole tokens, which helps short queries a lot
        grams.extend(_TOKEN_RE.findall(text))
        return grams

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for gram in self._ngrams(text):
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])


class OpenAIEmbedder(Embedder):
    """text-embedding-3-small via the OpenAI API. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        self.model = model
        self.dim = dim
        try:
            from openai import OpenAI  # local import: optional dependency
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class VoyageEmbedder(Embedder):
    """voyage-3-lite via Voyage AI (Anthropic's recommended embeddings
    partner, since Anthropic does not serve its own embeddings API).
    Requires VOYAGE_API_KEY.
    """

    def __init__(self, model: str = "voyage-3-lite", dim: int = 512):
        self.model = model
        self.dim = dim
        try:
            import voyageai  # local import: optional dependency
        except ImportError as e:
            raise ImportError(
                "voyageai package not installed. Run: pip install voyageai"
            ) from e
        self._client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

    def embed(self, texts: list[str]) -> np.ndarray:
        result = self._client.embed(texts, model=self.model, input_type="document")
        vecs = np.array(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def get_embedder(provider: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider!r}")
