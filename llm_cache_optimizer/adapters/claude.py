"""Anthropic Claude adapter with explicit cache_control breakpoints.

Claude can cache reusable content through ``cache_control`` markers on
cacheable blocks.  This adapter keeps stable prompt layers in the native
Claude ``system`` parameter and marks that system block when it is large
enough to be cacheable.

Usage::

    from llm_cache_optimizer import CacheAwareClaude

    client = CacheAwareClaude(api_key="sk-ant-...")
    client.add_core("You are a helpful coding assistant. ..." * 10)
    client.add_static_context("Project docs here ...")

    response = client.chat("Explain decorators.")
    print(client.cache_report())
"""

from __future__ import annotations

from typing import Any

from ..client import CacheAwareClient
from ..metrics import CacheMetrics
from ..serializer import CanonicalSerializer


class CacheAwareClaude(CacheAwareClient):
    """Cache-aware wrapper around ``anthropic.Anthropic().messages.create``.

    Unlike OpenAI / DeepSeek (automatic prefix caching), Claude requires
    **explicit** ``cache_control`` breakpoints on stable content blocks.
    This adapter automatically converts stable prompt layers into
    cacheable content blocks with ``{"type": "ephemeral"}`` markers.

    Minimum block size for caching: **1024 tokens** (≈ 4 000 characters).
    If a layer is too small, the adapter merges adjacent stable layers
    into a single cacheable block.
    """

    #: Approximate characters per token (used for size estimation).
    _CHARS_PER_TOKEN: float = 4.0

    #: Minimum characters for a cacheable block (1024 tokens × 4 chars).
    _MIN_CACHEABLE_CHARS: int = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        client: Any | None = None,
        serializer: CanonicalSerializer | None = None,
        metrics: CacheMetrics | None = None,
        max_tokens: int = 4096,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.anthropic_client = client or self._build_client(
            api_key=api_key,
            **client_kwargs,
        )
        super().__init__(
            chat_callable=self._chat_completion,
            serializer=serializer,
            metrics=metrics or CacheMetrics.from_provider(model),
            usage_extractor=_extract_usage,
            assistant_text_extractor=_extract_assistant_text,
        )

    def _build_client(self, api_key: str | None, **client_kwargs: Any) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "CacheAwareClaude requires the Anthropic SDK. "
                "Install it with `pip install anthropic`."
            ) from exc

        kwargs = dict(client_kwargs)
        if api_key is not None:
            kwargs["api_key"] = api_key
        return Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Message construction with cache_control breakpoints
    # ------------------------------------------------------------------

    def _split_claude_request(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Convert provider-neutral messages into Claude system/messages fields.

        Strategy:
        1. Group the leading stable system messages into Claude's system field.
        2. If the merged system block is large enough, add cache_control.
        3. Dynamic user / assistant / tool messages pass through as messages.
        """

        if not messages:
            return None, []

        # Separate system (stable) prefix from the rest
        stable_parts: list[str] = []
        dynamic_messages: list[dict[str, str]] = []
        found_dynamic = False

        for msg in messages:
            if msg["role"] == "system" and not found_dynamic:
                stable_parts.append(self.serializer.normalize(msg["content"]))
            else:
                found_dynamic = True
                dynamic_messages.append(msg)

        system_blocks: list[dict[str, Any]] | None = None

        if stable_parts:
            merged_text = "\n\n".join(stable_parts)

            block: dict[str, Any] = {"type": "text", "text": merged_text}
            if len(merged_text) >= self._MIN_CACHEABLE_CHARS:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks = [block]

        # Convert dynamic messages
        claude_messages: list[dict[str, Any]] = []
        for msg in dynamic_messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # Late system messages → inject as user content
                claude_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                })
            elif role == "tool":
                # Tool results → Claude uses "user" role with tool_result blocks
                claude_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content,
                        }
                    ],
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant with tool calls
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tc.get("function", {}).get("arguments", {}),
                    })
                claude_messages.append({"role": "assistant", "content": blocks})
            else:
                claude_messages.append({
                    "role": role,
                    "content": content if isinstance(content, str) else str(content),
                })

        return system_blocks, claude_messages

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        system_blocks, claude_messages = self._split_claude_request(messages)

        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("model", self.model)
        request_kwargs.setdefault("max_tokens", self.max_tokens)
        if system_blocks:
            request_kwargs["system"] = system_blocks
        request_kwargs["messages"] = claude_messages

        return self.anthropic_client.messages.create(**request_kwargs)

    def chat(self, user_input: str, **kwargs: Any) -> Any:
        """Send a cache-aware chat request through the Anthropic Messages API."""

        return super().chat(user_input, **kwargs)


def _extract_usage(response: Any) -> Any:
    """Extract usage from Anthropic response format."""
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_assistant_text(response: Any) -> str | None:
    """Extract text from Anthropic response content blocks."""
    if isinstance(response, dict):
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text")
        return response.get("text")

    content = getattr(response, "content", None)
    if isinstance(content, list):
        for block in content:
            if getattr(block, "type", None) == "text":
                return getattr(block, "text", None)
    return getattr(response, "text", None)
