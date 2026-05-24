from llm_cache_optimizer import CacheAwareOpenAI


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return {
            "choices": [{"message": {"content": "adapter ok"}}],
            "usage": {
                "prompt_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 150},
                "completion_tokens": 20,
            },
        }


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


def test_cache_aware_openai_sends_cache_ordered_messages():
    fake = FakeOpenAIClient()
    client = CacheAwareOpenAI(client=fake, model="test-model")
    client.add_core("core")
    client.add_static_context({"b": 2, "a": 1, "created_at": "2026-05-24T10:00:00Z"})

    response = client.chat("hello", temperature=0)

    assert response["choices"][0]["message"]["content"] == "adapter ok"
    assert fake.chat.completions.last_request == {
        "model": "test-model",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "core"},
            {"role": "system", "content": '{"a":1,"b":2}'},
            {"role": "user", "content": "hello"},
        ],
    }
    assert client.metrics.hit_rate == 0.75
