"""Provider adapters for cache-aware LLM clients."""

from .openai import CacheAwareOpenAI

__all__ = ["CacheAwareOpenAI"]
