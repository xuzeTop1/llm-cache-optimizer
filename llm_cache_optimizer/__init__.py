"""Runtime primitives for cache-aware LLM agents."""

from .client import CacheAwareClient
from .adapters.claude import CacheAwareClaude
from .adapters.openai import CacheAwareOpenAI
from .layers import PromptBuilder, PromptLayer, PromptLayers
from .memory import SessionMemory
from .metrics import CacheMetrics
from .serializer import CanonicalSerializer

__all__ = [
    "CacheAwareClient",
    "CacheAwareClaude",
    "CacheAwareOpenAI",
    "CacheMetrics",
    "CanonicalSerializer",
    "PromptBuilder",
    "PromptLayer",
    "PromptLayers",
    "SessionMemory",
]

__version__ = "0.3.0"
