# LLM Cache Optimizer

Agent Cache Runtime for building cache-aware LLM applications.

`llm-cache-optimizer` helps agent developers keep prompt prefixes stable, canonicalize volatile inputs, track cached-token savings, and compact long conversations into reusable session memory.

It works with OpenAI-style chat messages today, with a first-class OpenAI SDK adapter and provider-neutral primitives for DeepSeek, Claude, Gemini, OpenAI-compatible APIs, Codex, Claude Code, and OpenCode workflows.

## Why This Exists

Most modern LLM providers use prefix caching. If the first part of a request matches a previously processed prompt prefix, the provider can reuse cached tokens and reduce latency and cost.

The hard part is not knowing that caching exists. The hard part is keeping your agent prompt stable across real multi-turn workflows.

This project gives you runtime building blocks for that job:

- Stable prompt layering: core system, tool schema, static context, session memory, history, runtime input
- Canonical serialization: sorted JSON keys, normalized whitespace, stripped timestamps and request IDs
- Metrics: cache hit rate, cached tokens, estimated saved cost
- Provider adapter: `CacheAwareOpenAI`
- Session memory: local summary and keyword extraction for cache-friendly history compaction

## Install

```bash
pip install llm-cache-optimizer
```

For the OpenAI adapter:

```bash
pip install "llm-cache-optimizer[openai]"
```

## Quick Start

```python
from llm_cache_optimizer import CacheAwareClient

client = CacheAwareClient()
client.add_core("You are a concise coding assistant.")
client.add_tool_schema({"name": "read_file", "description": "Read a workspace file."})
client.add_static_context("Project: llm-cache-optimizer")

messages = client.chat("Show me the cache-aware message layout.")
print(messages)
```

The stable layers are always placed before dynamic user input:

```text
core_system -> tool_schema -> static_context -> session_memory -> history -> runtime
```

## OpenAI Adapter

```python
from llm_cache_optimizer import CacheAwareOpenAI

client = CacheAwareOpenAI(api_key="...", model="gpt-4o-mini")
client.add_core("You are a concise coding assistant.")
client.add_static_context("Stable project docs go here.")

response = client.chat("Refactor this function.")
print(client.cache_report())
```

The adapter wraps:

```python
OpenAI().chat.completions.create(...)
```

It automatically builds cache-ordered messages and reads common usage fields such as:

- `usage.prompt_tokens`
- `usage.prompt_tokens_details.cached_tokens`
- `usage.completion_tokens`

## DeepSeek

DeepSeek is a useful provider for cache benchmarking because it has automatic prefix caching, a low minimum cacheable prefix size of about 64 tokens, and cached input tokens are roughly 10% of normal input cost.

Use the OpenAI-compatible adapter with DeepSeek's `base_url`:

```python
from llm_cache_optimizer import CacheAwareOpenAI

client = CacheAwareOpenAI(
    api_key="sk-xxx",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
)

client.add_core(
    "You are a helpful coding assistant. Keep this stable and long enough "
    "for DeepSeek prefix caching."
)
client.add_static_context("Project docs here...")

for question in ["Explain decorators", "Show a retry pattern", "Write a cache layer"]:
    response = client.chat(question)
    print(client.cache_report())
```

DeepSeek-compatible usage objects may expose cache fields such as:

- `usage.prompt_cache_hit_tokens`
- `usage.prompt_cache_miss_tokens`

`CacheMetrics.update_from_usage()` reads `prompt_cache_hit_tokens` when present.

## Session Memory

Long-running agents should not keep rebuilding unstable prompt prefixes. Use session memory to summarize history and extract reusable keywords.

```python
from llm_cache_optimizer import CacheAwareClient

client = CacheAwareClient()
client.add_core("You are a cache-aware agent runtime.")
client.chat("We are turning this repo into a Python runtime.")
client.chat("Next we need an OpenAI adapter and a memory demo.")

memory = client.refresh_memory()
print(memory["summary"])
print(memory["keywords"])
```

Example output:

```python
{
    "summary": "user: We are turning this repo into a Python runtime. user: Next we need an OpenAI adapter and a memory demo.",
    "keywords": ["adapter", "memory", "openai", "runtime"],
}
```

## Core API

```python
from llm_cache_optimizer import (
    CacheAwareClient,
    CacheAwareOpenAI,
    CacheMetrics,
    CanonicalSerializer,
    PromptBuilder,
    SessionMemory,
)
```

### CanonicalSerializer

```python
from llm_cache_optimizer import CanonicalSerializer

serializer = CanonicalSerializer()

stable = serializer.normalize({
    "b": 2,
    "a": "hello    world",
    "created_at": "2026-05-24T10:00:00Z",
    "request_id": "req_123456789",
})

print(stable)
# {"a":"hello world","b":2}
```

### PromptBuilder

```python
from llm_cache_optimizer import PromptBuilder

builder = PromptBuilder()
builder.add_core("You are a helpful assistant.")
builder.add_tool_schema({"name": "read_file"})
builder.add_static_context("Stable docs")
builder.add_history("Earlier user message")
builder.add_runtime("Current user message")

messages = builder.build()
```

### CacheMetrics

```python
from llm_cache_optimizer import CacheMetrics

metrics = CacheMetrics(input_cost_per_1m=2.50, cached_input_cost_per_1m=1.25)
metrics.update_from_usage({
    "prompt_tokens": 1200,
    "prompt_tokens_details": {"cached_tokens": 900},
    "completion_tokens": 32,
})

print(metrics.report())
```

## Examples

- [`examples/basic.py`](./examples/basic.py): minimal runtime example
- [`examples/memory_demo.py`](./examples/memory_demo.py): local summary and keyword extraction
- [`examples/multi_provider_example.py`](./examples/multi_provider_example.py): provider-oriented agent loop patterns
- [`examples/opencode_example.py`](./examples/opencode_example.py): OpenCode pattern
- [`examples/claude_code_example.py`](./examples/claude_code_example.py): Claude Code pattern
- [`examples/codex_example.py`](./examples/codex_example.py): Codex pattern

## Current Roadmap

- v0.1.0: package structure, serializer, prompt layers, cache-aware client
- v0.2.0: OpenAI adapter, metrics, local session memory
- v0.3.0: benchmark system for baseline vs optimized agent loops
- v0.4.0: DeepSeek prefix diagnostics and provider-specific optimization reports
- Future: Claude cache-control adapter, OpenCode hook, Claude Code skill, Codex middleware

## Cache-Aware Design Rules

- Put stable content first.
- Never rebuild the system prompt per turn.
- Keep tool schemas and static context in stable layers.
- Put runtime data, timestamps, retrieved chunks, and user input near the end.
- Normalize tool outputs before appending them to history.
- Compact long history into session memory instead of moving the prefix.

## License

MIT
