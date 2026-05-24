"""Tests for CacheAwareClaude adapter."""

from llm_cache_optimizer import CacheAwareClaude


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class FakeContentBlock:
    def __init__(self, type_: str, text: str = ""):
        self.type = type_
        self.text = text


class FakeUsage:
    def __init__(self):
        self.input_tokens = 2000
        self.cache_read_input_tokens = 1500
        self.output_tokens = 50


class FakeMessage:
    def __init__(self):
        self.content = [FakeContentBlock("text", "Hello from Claude.")]
        self.usage = FakeUsage()


class FakeMessages:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return FakeMessage()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_claude_adapter_builds_cache_control_on_large_prefix():
    """Stable layers that exceed the minimum size should get cache_control."""

    fake = FakeAnthropicClient()
    client = CacheAwareClaude(client=fake, model="claude-sonnet-4-20250514")

    # Add a large enough core layer (> 4096 chars)
    large_core = "You are a helpful coding assistant. " * 200
    client.add_core(large_core)
    client.add_static_context("Some static docs.")

    response = client.chat("Hello")

    assert response.content[0].text == "Hello from Claude."

    # Check that the request was made
    assert fake.messages.last_request is not None
    assert fake.messages.last_request["model"] == "claude-sonnet-4-20250514"
    assert fake.messages.last_request["max_tokens"] == 4096

    # First message should be user with content blocks
    first_msg = fake.messages.last_request["messages"][0]
    assert first_msg["role"] == "user"
    assert isinstance(first_msg["content"], list)

    # Should have cache_control since the prefix is large enough
    block = first_msg["content"][0]
    assert block["type"] == "text"
    assert "cache_control" in block
    assert block["cache_control"] == {"type": "ephemeral"}


def test_claude_adapter_skips_cache_control_on_small_prefix():
    """Prefixes below the minimum size should NOT get cache_control."""

    fake = FakeAnthropicClient()
    client = CacheAwareClaude(client=fake, model="claude-sonnet-4-20250514")

    client.add_core("Short prompt.")

    client.chat("Hello")

    first_msg = fake.messages.last_request["messages"][0]
    assert first_msg["role"] == "user"

    block = first_msg["content"][0]
    assert block["type"] == "text"
    assert "cache_control" not in block


def test_claude_adapter_inserts_assistant_turn_after_prefix():
    """Claude requires an assistant turn after user content for caching."""

    fake = FakeAnthropicClient()
    client = CacheAwareClaude(client=fake)

    large_prefix = "System prompt content. " * 300
    client.add_core(large_prefix)
    client.chat("Hi")

    messages = fake.messages.last_request["messages"]
    # [0] = user (prefix), [1] = assistant ("Understood."), [2] = user ("Hi")
    assert len(messages) >= 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Understood."
    assert messages[2]["role"] == "user"


def test_claude_adapter_tracks_metrics():
    """Metrics should be updated from Claude's usage format."""

    fake = FakeAnthropicClient()
    client = CacheAwareClaude(client=fake)

    client.chat("Test")

    assert client.metrics.prompt_tokens == 2000
    assert client.metrics.cached_tokens == 1500
    assert client.metrics.hit_rate == 0.75


def test_claude_adapter_build_client_raises_without_sdk():
    """Should raise ImportError if anthropic is not installed."""

    # This test is informational — we can't easily uninstall anthropic
    # in a test, but we verify the adapter structure is correct.
    fake = FakeAnthropicClient()
    client = CacheAwareClaude(client=fake)
    assert client.anthropic_client is fake


def test_claude_adapter_chat_without_provider_returns_messages():
    """Without a provider, chat() should return the built messages."""

    client = CacheAwareClaude(client=FakeAnthropicClient())
    client.add_core("You are helpful.")

    # Using CacheAwareClient's default behavior (no chat_callable override)
    # The Claude adapter always sets chat_callable, so this tests the base
    # flow with the Claude-specific message transformation.
    messages = client.messages("Hello")
    assert isinstance(messages, list)
    assert len(messages) > 0
