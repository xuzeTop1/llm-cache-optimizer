"""Prompt layering primitives for stable prefix construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .serializer import CanonicalSerializer

Role = Literal["system", "user", "assistant", "tool"]
LayerName = Literal[
    "core_system",
    "tool_schema",
    "static_context",
    "session_memory",
    "history",
    "runtime",
]


@dataclass(frozen=True)
class PromptLayer:
    """A named prompt layer with ordered, cache-aware content."""

    name: LayerName
    content: Any
    role: Role = "system"
    stable: bool = True


@dataclass
class PromptLayers:
    """Container for the six recommended cache-aware prompt layers."""

    core_system: list[PromptLayer] = field(default_factory=list)
    tool_schema: list[PromptLayer] = field(default_factory=list)
    static_context: list[PromptLayer] = field(default_factory=list)
    session_memory: list[PromptLayer] = field(default_factory=list)
    history: list[PromptLayer] = field(default_factory=list)
    runtime: list[PromptLayer] = field(default_factory=list)

    @property
    def ordered(self) -> list[PromptLayer]:
        """Return layers in cache-friendly order, with runtime content last."""

        return [
            *self.core_system,
            *self.tool_schema,
            *self.static_context,
            *self.session_memory,
            *self.history,
            *self.runtime,
        ]


class PromptBuilder:
    """Build OpenAI-style messages while keeping stable content first."""

    def __init__(self, serializer: CanonicalSerializer | None = None) -> None:
        self.serializer = serializer or CanonicalSerializer()
        self.layers = PromptLayers()

    def add_core(self, content: Any, role: Role = "system") -> "PromptBuilder":
        return self.add("core_system", content, role=role, stable=True)

    def add_tool_schema(self, content: Any, role: Role = "system") -> "PromptBuilder":
        return self.add("tool_schema", content, role=role, stable=True)

    def add_static_context(self, content: Any, role: Role = "system") -> "PromptBuilder":
        return self.add("static_context", content, role=role, stable=True)

    def add_session_memory(self, content: Any, role: Role = "system") -> "PromptBuilder":
        return self.add("session_memory", content, role=role, stable=False)

    def add_history(self, content: Any, role: Role = "user") -> "PromptBuilder":
        return self.add("history", content, role=role, stable=False)

    def add_runtime(self, content: Any, role: Role = "user") -> "PromptBuilder":
        return self.add("runtime", content, role=role, stable=False)

    def add(
        self,
        name: LayerName,
        content: Any,
        role: Role = "system",
        stable: bool | None = None,
    ) -> "PromptBuilder":
        layer = PromptLayer(
            name=name,
            content=content,
            role=role,
            stable=name in {"core_system", "tool_schema", "static_context"}
            if stable is None
            else stable,
        )
        getattr(self.layers, name).append(layer)
        return self

    def extend_history(self, messages: Iterable[dict[str, Any]]) -> "PromptBuilder":
        for message in messages:
            self.add_history(message.get("content", ""), role=message.get("role", "user"))
        return self

    def build(self) -> list[dict[str, str]]:
        """Build a provider-neutral list of role/content messages."""

        messages: list[dict[str, str]] = []
        for layer in self.layers.ordered:
            content = self.serializer.normalize(layer.content)
            if not content:
                continue
            messages.append({"role": layer.role, "content": content})
        return messages

    def stable_prefix(self) -> list[dict[str, str]]:
        """Build only layers intended to stay stable across calls."""

        messages: list[dict[str, str]] = []
        for layer in self.layers.ordered:
            if not layer.stable:
                continue
            content = self.serializer.normalize(layer.content)
            if content:
                messages.append({"role": layer.role, "content": content})
        return messages
