"""Per-query cost and latency tracking, so the app can report what each
answer actually cost instead of hiding it (real deployments need this)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CONFIG


@dataclass
class QueryStats:
    provider: str = "mock"
    model: str = "mock"
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def cost_usd(self) -> float:
        key = f"{self.provider}:{self.model}"
        rates = CONFIG.pricing.get(key)
        if not rates:
            return 0.0
        return (self.input_tokens / 1000) * rates["input"] + \
               (self.output_tokens / 1000) * rates["output"]

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "cost_usd": round(self.cost_usd(), 6),
        }
