"""Five-layer cache-aware agent loop using the runtime library.

This example works with DeepSeek, OpenAI, OpenRouter, and other OpenAI-compatible
providers. The library handles stable prompt ordering, canonical serialization,
history tracking, session memory, and cache metrics.
"""

from __future__ import annotations

import os
from typing import Any

from llm_cache_optimizer import CacheAwareOpenAI, SessionMemory


LAYER_CORE = """
You are a helpful coding assistant. Follow these principles:

1. Code quality: type hints, docstrings, error handling, tests
2. Architecture: composition over inheritance, single responsibility
3. Communication: be concise, show code over description
4. Safety: never delete files without confirmation, explain risks
"""

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["cmd"],
            },
        },
    },
]

LAYER_RAG = """
<project_context>
Language: Python 3.9+
Build: hatchling
Testing: pytest
Purpose: cache-aware LLM agent runtime
</project_context>
"""


def build_client() -> CacheAwareOpenAI:
    """Create a cache-aware OpenAI-compatible client from environment variables."""

    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-xxx")
        model = "deepseek-chat"
        base_url = "https://api.deepseek.com/v1"
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "sk-xxx")
        model = os.environ.get("MODEL", "anthropic/claude-sonnet-4")
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "sk-xxx")
        model = os.environ.get("MODEL", "gpt-4o-mini")
        base_url = os.environ.get("API_BASE_URL")

    client = CacheAwareOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        memory=SessionMemory(max_keywords=10),
    )
    client.add_core(LAYER_CORE)
    client.add_tool_schema(TOOL_SCHEMAS)
    client.add_static_context(LAYER_RAG)
    return client


def main() -> None:
    """Run a multi-turn demo with stable prompt layers and metrics reporting."""

    client = build_client()
    questions = [
        "Explain how the prompt cache runtime should layer system prompts.",
        "Show a retry pattern for provider calls.",
        "Summarize the session memory and suggest the next implementation step.",
    ]

    for question in questions:
        response = client.chat(question)
        print(response.choices[0].message.content)
        client.refresh_memory()
        print(client.memory_report())
        print(client.cache_report())


if __name__ == "__main__":
    main()
