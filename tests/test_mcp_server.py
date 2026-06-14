from llm_cache_optimizer.mcp_server import MCPServer


def test_optimize_prompt_preserves_system_messages_without_static_prefix():
    server = MCPServer()

    result = server._optimize_prompt(
        {
            "messages": [
                {"role": "system", "content": "Stable system prompt."},
                {"role": "user", "content": "Hello"},
            ],
        }
    )

    assert result["optimized_messages"] == [
        {"role": "system", "content": "Stable system prompt."},
        {"role": "user", "content": "Hello"},
    ]


def test_canonicalize_output_strip_fields_are_per_call():
    server = MCPServer()

    first = server._canonicalize_output(
        {
            "data": {"keep": 1, "session_id": "abc"},
            "strip_fields": ["session_id"],
        }
    )
    second = server._canonicalize_output(
        {
            "data": {"keep": 1, "session_id": "abc"},
        }
    )

    assert first["canonicalized"] == '{"keep":1}'
    assert second["canonicalized"] == '{"keep":1,"session_id":"abc"}'


def test_cache_report_counts_contiguous_stable_prefix():
    server = MCPServer()

    result = server._cache_report(
        {
            "provider": "openai",
            "messages": [
                {"role": "system", "content": "a" * 100},
                {"role": "developer", "content": "b" * 100},
                {"role": "user", "content": "dynamic"},
                {"role": "system", "content": "late system is dynamic"},
            ],
        }
    )

    assert result["stable_prefix_tokens"] == 50
