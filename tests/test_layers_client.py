from llm_cache_optimizer import CacheAwareClient, CacheMetrics, PromptBuilder


def test_prompt_builder_orders_dynamic_content_last():
    builder = PromptBuilder()
    builder.add_runtime("runtime")
    builder.add_core("core")
    builder.add_history("history")
    builder.add_static_context("static")

    assert builder.build() == [
        {"role": "system", "content": "core"},
        {"role": "system", "content": "static"},
        {"role": "user", "content": "history"},
        {"role": "user", "content": "runtime"},
    ]


def test_cache_aware_client_returns_messages_without_provider():
    client = CacheAwareClient()
    client.add_core("core")

    assert client.chat("hello") == [
        {"role": "system", "content": "core"},
        {"role": "user", "content": "hello"},
    ]


def test_cache_aware_client_updates_metrics_from_provider_usage():
    def fake_provider(_messages, **_kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens": 10,
            },
        }

    client = CacheAwareClient(chat_callable=fake_provider)
    client.add_core("core")
    response = client.chat("hello")

    assert response["choices"][0]["message"]["content"] == "ok"
    assert client.metrics.hit_rate == 0.8


def test_cache_metrics_from_provider_estimates_savings():
    metrics = CacheMetrics.from_provider("gpt-4o")
    metrics.update_from_usage(
        {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    )

    assert metrics.input_cost_per_1m == 2.5
    assert metrics.cached_input_cost_per_1m == 1.25
    assert metrics.estimated_cost_saved == 0.001
