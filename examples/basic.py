"""Minimal cache-aware runtime example."""

from llm_cache_optimizer import CacheAwareClient


def fake_provider(messages, **_kwargs):
    return {
        "choices": [{"message": {"content": "Hello from a cache-aware client."}}],
        "usage": {
            "prompt_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens": 32,
        },
    }


client = CacheAwareClient(chat_callable=fake_provider)
client.add_core("You are a concise coding assistant.")
client.add_tool_schema(
    {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "created_at": "2026-05-24T10:00:00Z",
    }
)
client.add_static_context("Project: llm-cache-optimizer")

response = client.chat("Show me the cache-aware message layout.")

print(response["choices"][0]["message"]["content"])
print(client.cache_report())
