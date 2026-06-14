"""High-level cache-aware client wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .layers import PromptBuilder
from .memory import SessionMemory
from .metrics import CacheMetrics
from .serializer import CanonicalSerializer

ChatCallable = Callable[[list[dict[str, str]]], Any]
UsageExtractor = Callable[[Any], Any]
AssistantTextExtractor = Callable[[Any], str | None]


class CacheAwareClient:
    """Build cache-friendly messages and optionally dispatch them to a provider."""

    def __init__(
        self,
        chat_callable: ChatCallable | None = None,
        serializer: CanonicalSerializer | None = None,
        metrics: CacheMetrics | None = None,
        memory: SessionMemory | None = None,
        usage_extractor: UsageExtractor | None = None,
        assistant_text_extractor: AssistantTextExtractor | None = None,
    ) -> None:
        self.chat_callable = chat_callable
        self.serializer = serializer or CanonicalSerializer()
        self.metrics = metrics or CacheMetrics()
        self.memory = memory or SessionMemory()
        self.usage_extractor = usage_extractor or _extract_usage
        self.assistant_text_extractor = assistant_text_extractor or _extract_assistant_text
        self.builder = PromptBuilder(serializer=self.serializer)
        self.history: list[dict[str, str]] = []
        self._session_memory: dict[str, Any] | None = None

    def add_core(self, content: Any) -> "CacheAwareClient":
        self.builder.add_core(content)
        return self

    def add_tool_schema(self, content: Any) -> "CacheAwareClient":
        self.builder.add_tool_schema(content)
        return self

    def add_static_context(self, content: Any) -> "CacheAwareClient":
        self.builder.add_static_context(content)
        return self

    def add_session_memory(self, content: Any) -> "CacheAwareClient":
        self.builder.add_session_memory(content)
        return self

    def messages(self, user_input: str | None = None) -> list[dict[str, str]]:
        """Build messages with stable layers first and optional user input last."""

        messages = self.builder.build()
        if self._session_memory is not None:
            messages.append(
                {
                    "role": "system",
                    "content": self.serializer.normalize(self._session_memory),
                }
            )
        messages.extend(self.history)
        if user_input is not None:
            messages.append({"role": "user", "content": self.serializer.normalize(user_input)})
        return messages

    def chat(self, user_input: str, **kwargs: Any) -> Any:
        """Append user input, call the provider if configured, and track usage."""

        messages = self.messages(user_input)
        self.history.append({"role": "user", "content": self.serializer.normalize(user_input)})

        if self.chat_callable is None:
            return messages

        response = self.chat_callable(messages, **kwargs)
        usage = self.usage_extractor(response)
        if usage is not None:
            self.metrics.update_from_usage(usage)

        assistant_text = self.assistant_text_extractor(response)
        if assistant_text:
            self.history.append({"role": "assistant", "content": assistant_text})

        return response

    def cache_report(self) -> str:
        return self.metrics.report()

    def refresh_memory(self) -> dict[str, Any]:
        """Summarize current history and expose it as a session-memory layer."""

        self.memory.update(self.history)
        self._session_memory = self.memory.to_layer()
        return self._session_memory

    def memory_report(self) -> str:
        """Return the current local memory summary and extracted keywords."""

        if self._session_memory is None:
            self.refresh_memory()
        return self.memory.to_text()


def _extract_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_assistant_text(response: Any) -> str | None:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")
        return response.get("content")

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        return getattr(message, "content", None)
    return getattr(response, "content", None)
