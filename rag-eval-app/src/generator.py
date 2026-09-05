"""
Generation layer: turns (query, retrieved chunks) into a cited answer.

Guardrails implemented here, deliberately at two layers:
  1. A *retrieval-side* guardrail in pipeline.py: if the best fused score is
     below CONFIG.min_relevance_score, we never call the LLM at all and
     return "I don't know" -- cheaper and more reliable than hoping the
     model declines gracefully.
  2. A *prompt-side* guardrail here: the system prompt explicitly instructs
     the model to say it doesn't know rather than guess, and to cite every
     claim with a [n] marker tied to the numbered context chunks. This
     doesn't guarantee the model won't hallucinate, which is exactly why
     eval/run_eval.py measures citation validity separately rather than
     trusting the prompt to have worked.
"""
from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import CONFIG
from .cost_tracker import QueryStats
from .retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question \
using ONLY the numbered context chunks provided below. Rules:

1. Every factual claim in your answer must end with a citation marker like [1] \
or [2] referring to the context chunk number it came from.
2. If the context does not contain enough information to answer, say exactly: \
"I don't have enough information in the documentation to answer that." Do not guess.
3. Be concise and direct. Do not repeat the question back.
4. If multiple chunks support one claim, cite all of them, e.g. [1][3].
"""


@dataclass
class GeneratedAnswer:
    text: str
    citations_used: list[int]  # chunk indices (1-based) actually cited in the text
    refused: bool
    stats: QueryStats


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {rc.chunk.doc_id}, section: {rc.chunk.section})\n{rc.chunk.text}")
    return "\n\n".join(parts)


def _extract_citations(text: str, n_chunks: int) -> list[int]:
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(c for c in found if 1 <= c <= n_chunks)


class LLMBackend(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        raise NotImplementedError


class MockBackend(LLMBackend):
    """Deterministic, dependency-free extractive stand-in for a real LLM.

    It does not "understand" the question -- it just returns the top
    context chunk's text with a citation, and triggers the refusal string
    if no chunks were passed. This exists purely so the *pipeline* (fusion,
    guardrails, citation extraction, eval harness) can be built, run, and
    tested end-to-end without an API key. Swap LLM_PROVIDER=anthropic to
    get a real model producing real synthesized answers.
    """
    provider = "mock"
    model = "mock"

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        m = re.search(r"\[1\] \(source.*?\)\n(.*?)(\n\n\[2\]|\Z)", user, re.DOTALL)
        input_tokens = len(system.split()) + len(user.split())
        if not m:
            text = "I don't have enough information in the documentation to answer that."
            return text, input_tokens, len(text.split())
        snippet = m.group(1).strip()
        sentences = re.split(r"(?<=[.!?])\s+", snippet)
        text = " ".join(sentences[:2]).strip() + " [1]"
        return text, input_tokens, len(text.split())


class AnthropicBackend(LLMBackend):
    provider = "anthropic"

    def __init__(self, model: str = CONFIG.anthropic_model):
        self.model = model
        try:
            import anthropic  # local import: optional dependency
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=CONFIG.max_answer_tokens,
            temperature=CONFIG.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class OpenAIBackend(LLMBackend):
    provider = "openai"

    def __init__(self, model: str = CONFIG.openai_model):
        self.model = model
        try:
            from openai import OpenAI  # local import: optional dependency
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=CONFIG.max_answer_tokens,
            temperature=CONFIG.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


def get_backend(provider: str) -> LLMBackend:
    if provider == "mock":
        return MockBackend()
    if provider == "anthropic":
        return AnthropicBackend()
    if provider == "openai":
        return OpenAIBackend()
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def generate_answer(query: str, chunks: list[RetrievedChunk], backend: LLMBackend) -> GeneratedAnswer:
    context_block = _build_context_block(chunks)
    user_prompt = f"Context:\n\n{context_block}\n\nQuestion: {query}"

    start = time.perf_counter()
    text, in_tok, out_tok = backend.complete(SYSTEM_PROMPT, user_prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000

    citations = _extract_citations(text, len(chunks))
    refused = "don't have enough information" in text.lower()

    stats = QueryStats(
        provider=backend.provider,
        model=backend.model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        generation_ms=elapsed_ms,
    )
    return GeneratedAnswer(text=text, citations_used=citations, refused=refused, stats=stats)
