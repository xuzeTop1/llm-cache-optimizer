"""Cache metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CacheMetrics:
    """Track cache effectiveness and estimated savings."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    input_cost_per_1m: float = 0.0
    cached_input_cost_per_1m: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    @property
    def token_saved(self) -> int:
        return self.cached_tokens

    @property
    def estimated_cost_saved(self) -> float:
        uncached = self.cached_tokens * self.input_cost_per_1m / 1_000_000
        cached = self.cached_tokens * self.cached_input_cost_per_1m / 1_000_000
        return max(0.0, uncached - cached)

    def update_from_usage(self, usage: object | dict) -> "CacheMetrics":
        """Merge common provider usage fields into this metric object."""

        prompt_tokens = _get(usage, "prompt_tokens", "input_tokens") or 0
        completion_tokens = _get(usage, "completion_tokens", "output_tokens") or 0
        cached_tokens = (
            _get(usage, "cached_tokens")
            or _get_nested(usage, "prompt_tokens_details", "cached_tokens")
            or _get(usage, "prompt_cache_hit_tokens")
            or _get(usage, "cache_read_input_tokens")
            or 0
        )

        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.cached_tokens += int(cached_tokens)
        return self

    def report(self) -> str:
        return "\n".join(
            [
                f"Cache hit: {self.hit_rate:.0%}",
                f"Tokens saved: {self.token_saved:,}",
                f"Cost saved: ${self.estimated_cost_saved:.2f}",
            ]
        )


def _get(value: object | dict, *names: str) -> object | None:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _get_nested(value: object | dict, parent: str, child: str) -> object | None:
    nested = _get(value, parent)
    if nested is None:
        return None
    return _get(nested, child)
