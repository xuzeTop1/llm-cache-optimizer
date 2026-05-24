"""OpenAI-compatible backend example for Codex, OpenCode, and custom gateways."""

from __future__ import annotations

import os

from llm_cache_optimizer import CacheAwareOpenAI


def build_client() -> CacheAwareOpenAI:
    """Create a cache-aware client for any OpenAI-compatible endpoint."""

    client = CacheAwareOpenAI(
        api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "sk-xxx")),
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("API_BASE_URL"),
    )
    client.add_core(
        "You are a helpful coding assistant. Use type hints, handle errors "
        "explicitly, and explain tradeoffs concisely."
    )
    client.add_tool_schema(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command and return output.",
                },
            },
        ]
    )
    client.add_static_context("Workspace: cache-aware coding assistant.")
    return client


def main() -> None:
    """Run a short OpenAI-compatible cache-aware chat loop."""

    client = build_client()
    for question in [
        "Help me refactor the auth module to use JWT.",
        "Now add tests for the token parser.",
    ]:
        response = client.chat(question)
        print(response.choices[0].message.content)
        print(client.cache_report())


if __name__ == "__main__":
    main()
