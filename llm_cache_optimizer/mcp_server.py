#!/usr/bin/env python3
"""
MCP Server for llm-cache-optimizer.

Exposes prompt optimization, canonicalization, session memory compression,
and cache metrics to any MCP-compatible client (OpenCode, Claude Code, etc.).

Protocol: JSON-RPC 2.0 over stdio (MCP 2024-11-05).

Usage::

    python -m llm_cache_optimizer.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .client import CacheAwareClient
from .layers import PromptBuilder
from .memory import SessionMemory
from .metrics import CacheMetrics
from .serializer import CanonicalSerializer


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "optimize_prompt",
        "description": (
            "Reorder and canonicalize messages for prefix caching. "
            "Returns a cache-friendly message list with stable layers first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "OpenAI-style chat messages to optimize.",
                },
                "provider": {
                    "type": "string",
                    "enum": ["openai", "deepseek", "anthropic"],
                    "default": "openai",
                },
                "static_prefix": {
                    "type": "object",
                    "description": "Stable layers: core, tool_schema, static_context.",
                },
            },
            "required": ["messages"],
        },
    },
    {
        "name": "canonicalize_output",
        "description": (
            "Normalize tool output JSON for cache stability. "
            "Sorts keys, strips timestamps and volatile fields, collapses whitespace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "description": "Any JSON-serializable value to canonicalize.",
                },
                "strip_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["created_at", "request_id", "timestamp"],
                },
            },
            "required": ["data"],
        },
    },
    {
        "name": "refresh_memory",
        "description": (
            "Compress conversation history into a stable session memory block. "
            "Returns summary, keywords, active files, decisions, and next step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "history": {
                    "type": "array",
                    "description": "Raw message history (role + content).",
                },
                "summarizer": {
                    "type": "string",
                    "description": "Summarizer strategy: 'local' or 'llm'.",
                    "default": "local",
                },
            },
            "required": ["history"],
        },
    },
    {
        "name": "cache_report",
        "description": (
            "Analyze a prompt and return estimated cache hit rate, "
            "stable prefix size, and cost savings tips."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array"},
                "provider": {"type": "string", "default": "deepseek"},
                "usage": {
                    "type": "object",
                    "description": "Provider usage response with cached_tokens fields.",
                },
            },
            "required": ["messages", "provider"],
        },
    },
]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class MCPServer:
    """Minimal JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self) -> None:
        self.serializer = CanonicalSerializer()
        self._call_count = 0

    # -- Protocol -----------------------------------------------------------

    async def run(self) -> None:
        """Read JSON-RPC messages from stdin, dispatch, write to stdout."""
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self._dispatch(req)
            except json.JSONDecodeError:
                resp = self._error(None, -32700, "Parse error")
            self._send(resp)

    def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        method: str = req.get("method", "")
        params: dict[str, Any] = req.get("params", {})
        req_id = req.get("id")

        if method == "initialize":
            return self._result(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "llm-cache-optimizer",
                    "version": "0.3.0",
                },
            })

        if method == "notifications/initialized":
            # Client ack — no response needed, but we send empty result
            return self._result(req_id, {})

        if method == "tools/list":
            return self._result(req_id, {"tools": TOOLS})

        if method == "tools/call":
            return self._call_tool(params, req_id)

        return self._error(req_id, -32601, f"Method not found: {method}")

    # -- Tool implementations -----------------------------------------------

    def _call_tool(self, params: dict[str, Any], req_id: Any) -> dict[str, Any]:
        name = params.get("name", "")
        args = params.get("arguments", {})
        self._call_count += 1

        try:
            if name == "optimize_prompt":
                data = self._optimize_prompt(args)
            elif name == "canonicalize_output":
                data = self._canonicalize_output(args)
            elif name == "refresh_memory":
                data = self._refresh_memory(args)
            elif name == "cache_report":
                data = self._cache_report(args)
            else:
                return self._error(req_id, -32602, f"Unknown tool: {name}")

            return self._result(req_id, {
                "content": [{
                    "type": "text",
                    "text": json.dumps(data, ensure_ascii=False, indent=2),
                }],
            })
        except Exception as exc:
            return self._error(req_id, -32603, str(exc))

    def _optimize_prompt(self, args: dict[str, Any]) -> dict[str, Any]:
        """Reorder messages: static prefix → session memory → history → runtime."""
        messages = args.get("messages", [])
        provider = args.get("provider", "openai")
        static = args.get("static_prefix", {})

        builder = PromptBuilder(serializer=self.serializer)

        if static.get("core"):
            builder.add_core(static["core"])
        if static.get("tool_schema"):
            builder.add_tool_schema(static["tool_schema"])
        if static.get("static_context"):
            builder.add_static_context(static["static_context"])

        # Separate stable (system) from dynamic messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                # Already handled via static_prefix; skip duplicates
                continue
            elif role in ("user", "assistant"):
                builder.add_history(content, role=role)

        optimized = builder.build()

        return {
            "optimized_messages": optimized,
            "provider": provider,
            "layer_count": len(optimized),
            "stable_prefix_tokens_estimate": sum(
                len(m.get("content", "")) for m in builder.stable_prefix()
            ) // 4,
        }

    def _canonicalize_output(self, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize data for cache stability."""
        data = args.get("data", {})

        # Update volatile keys if custom strip_fields provided
        strip_fields = args.get("strip_fields")
        if strip_fields:
            self.serializer.volatile_keys.update(strip_fields)

        canonical = self.serializer.normalize(data)

        return {
            "canonicalized": canonical,
            "original_type": type(data).__name__,
            "char_count": len(canonical),
        }

    def _refresh_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Compress history into session memory block."""
        history = args.get("history", [])

        memory = SessionMemory()
        memory.update(history)

        summary = memory.summary
        keywords = memory.keywords

        # Extract active files and decisions from history heuristically
        active_files = _extract_files(history)
        decisions = _extract_decisions(history)
        next_step = _infer_next_step(history)

        return {
            "summary": summary,
            "keywords": keywords,
            "active_files": active_files,
            "decisions": decisions,
            "next_step": next_step,
        }

    def _cache_report(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze prompt for cache efficiency."""
        messages = args.get("messages", [])
        provider = args.get("provider", "deepseek")
        usage = args.get("usage")

        metrics = CacheMetrics.from_provider(provider)
        if usage:
            metrics.update_from_usage(usage)

        total_chars = sum(len(m.get("content", "")) for m in messages)
        total_tokens_est = total_chars // 4

        # Stable prefix = first 2 messages (system + static context)
        stable_chars = sum(
            len(m.get("content", "")) for m in messages[:2]
        ) if len(messages) > 2 else 0
        stable_tokens_est = stable_chars // 4

        hit_rate = round(stable_tokens_est / max(total_tokens_est, 1), 2)

        return {
            "provider": provider,
            "total_prompt_tokens": total_tokens_est,
            "stable_prefix_tokens": stable_tokens_est,
            "estimated_hit_rate": hit_rate,
            "cached_tokens_actual": metrics.cached_tokens,
            "cost_saved_usd": round(metrics.estimated_cost_saved, 4),
            "savings_tip": (
                "Keep system prompt and tool schemas in first 2 messages "
                "to maximize prefix caching."
            ),
        }

    # -- Helpers ------------------------------------------------------------

    def _result(self, req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def _send(self, msg: dict[str, Any]) -> None:
        print(json.dumps(msg, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# Heuristic extractors for refresh_memory
# ---------------------------------------------------------------------------


def _extract_files(history: list[dict[str, Any]]) -> list[str]:
    """Find file paths mentioned in conversation history."""
    import re
    text = " ".join(m.get("content", "") for m in history)
    # Match common file path patterns
    patterns = re.findall(
        r"(?:[\w/.-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|md|toml|cfg|ini))",
        text,
    )
    return list(dict.fromkeys(patterns))[:10]  # deduplicate, keep order, limit 10


def _extract_decisions(history: list[dict[str, Any]]) -> list[str]:
    """Find sentences that look like decisions."""
    decisions = []
    keywords = ("decided", "chose", "switched to", "using", "will use", "replaced")
    for msg in history:
        content = msg.get("content", "").lower()
        for kw in keywords:
            if kw in content:
                # Take the sentence containing the keyword
                for sentence in msg.get("content", "").split("."):
                    if kw in sentence.lower():
                        decisions.append(sentence.strip())
                        break
                break
    return decisions[:5]


def _infer_next_step(history: list[dict[str, Any]]) -> str:
    """Guess the most logical next action from context."""
    if not history:
        return "Start a conversation to build context."

    last_msg = history[-1].get("content", "").lower()

    if "test" in last_msg:
        return "Run the test suite to verify changes."
    if "refactor" in last_msg:
        return "Apply the refactoring and run tests."
    if "error" in last_msg or "bug" in last_msg:
        return "Debug the issue and fix the root cause."
    if "review" in last_msg:
        return "Review the changes and provide feedback."

    return "Continue with the next logical step."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server."""
    server = MCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
