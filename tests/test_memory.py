from llm_cache_optimizer import CacheAwareClient, SessionMemory


def test_session_memory_extracts_summary_and_keywords():
    memory = SessionMemory(max_keywords=4)

    memory.update(
        [
            {
                "role": "user",
                "content": "Build an OpenAI adapter for prompt cache metrics.",
            },
            {
                "role": "assistant",
                "content": "The OpenAI adapter should report cached tokens and cache hit rate.",
            },
        ]
    )

    assert "OpenAI adapter" in memory.summary or "openai adapter" in memory.summary.lower()
    assert "adapter" in memory.keywords
    assert "openai" in memory.keywords


def test_cache_aware_client_injects_refreshed_memory_before_history():
    client = CacheAwareClient()
    client.add_core("core")
    client.chat("Build a Python runtime for prompt cache optimization.")
    client.refresh_memory()

    messages = client.messages("What should we do next?")

    assert messages[0] == {"role": "system", "content": "core"}
    assert "keywords" in messages[1]["content"]
    assert messages[2] == {
        "role": "user",
        "content": "Build a Python runtime for prompt cache optimization.",
    }
    assert messages[-1] == {"role": "user", "content": "What should we do next?"}
