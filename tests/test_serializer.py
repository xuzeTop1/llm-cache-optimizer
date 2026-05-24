from llm_cache_optimizer import CanonicalSerializer


def test_normalize_removes_volatile_fields_and_sorts_keys():
    serializer = CanonicalSerializer()

    result = serializer.normalize(
        {
            "b": 2,
            "created_at": "2026-05-24T10:00:00Z",
            "a": "hello    world",
            "request_id": "req_123456789",
        }
    )

    assert result == '{"a":"hello world","b":2}'


def test_normalize_replaces_timestamps_and_uuids_in_text():
    serializer = CanonicalSerializer()

    result = serializer.normalize(
        "run 550e8400-e29b-41d4-a716-446655440000 at 2026-05-24T10:00:00Z"
    )

    assert result == "run <uuid> at <timestamp>"
