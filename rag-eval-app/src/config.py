"""
Central configuration for the RAG pipeline.

Everything that's a "knob" lives here so it's easy to point to in a README
or a design doc: chunk size, retrieval fan-out, model choice, etc.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
EVAL_SET_PATH = ROOT_DIR / "data" / "eval" / "eval_set.json"
INDEX_DIR = ROOT_DIR / "data" / "index"


@dataclass
class Config:
    # --- Chunking ---
    chunk_size_tokens: int = 220          # target chunk size (approx tokens, whitespace-based)
    chunk_overlap_tokens: int = 40        # sliding-window overlap between adjacent chunks

    # --- Retrieval ---
    top_k_dense: int = 8                  # candidates pulled from vector search
    top_k_sparse: int = 8                 # candidates pulled from BM25
    top_k_final: int = 4                  # chunks actually sent to the LLM after fusion
    rrf_k: int = 60                       # reciprocal-rank-fusion constant (standard default)

    # --- Generation ---
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_PROVIDER", "hash")
    )
    # "hash"   -> deterministic offline embedding, no API/network needed (default, used for
    #             local dev + this repo's own CI/tests so nothing here depends on a live key)
    # "openai" -> text-embedding-3-small via OpenAI API (needs OPENAI_API_KEY)
    # "voyage" -> voyage-3-lite via Voyage AI, Anthropic's recommended embeddings partner
    #             (needs VOYAGE_API_KEY)

    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock")
    )
    # "mock"      -> deterministic extractive stand-in, no API/network needed
    # "anthropic" -> claude-sonnet-4-6 via ANTHROPIC_API_KEY
    # "openai"    -> gpt-4o-mini via OPENAI_API_KEY

    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"
    max_answer_tokens: int = 500
    temperature: float = 0.0

    # --- Guardrails ---
    # RRF scores are small by construction (~1/rrf_k per list a chunk appears
    # in near rank 0, i.e. ~0.016-0.033 for the defaults above), so this
    # threshold is tuned relative to rrf_k, not an absolute "confidence".
    # Tightened/loosened based on eval/run_eval.py refusal-accuracy results.
    min_relevance_score: float = 0.012    # below this, we refuse to answer ("I don't know")

    # --- Cost tracking (USD per 1K tokens, update if pricing changes) ---
    pricing: dict = field(default_factory=lambda: {
        "anthropic:claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "openai:gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    })


CONFIG = Config()
