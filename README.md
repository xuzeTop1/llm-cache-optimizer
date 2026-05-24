# LLM Cache Optimizer

![Tests](https://github.com/xuzeTop1/llm-cache-optimizer/actions/workflows/test.yml/badge.svg)

Agent Cache Runtime for building cache-aware LLM applications.

`llm-cache-optimizer` helps agent developers keep prompt prefixes stable, canonicalize volatile inputs, track cached-token savings, and compact long conversations into reusable session memory.

It works with OpenAI-style chat messages today, with first-class adapters for OpenAI / DeepSeek and Anthropic Claude, plus provider-neutral primitives for Gemini, OpenAI-compatible APIs, Codex, Claude Code, and OpenCode workflows.

## Why This Exists

Most modern LLM providers use prefix caching. If the first part of a request matches a previously processed prompt prefix, the provider can reuse cached tokens and reduce latency and cost.

The hard part is not knowing that caching exists. The hard part is keeping your agent prompt stable across real multi-turn workflows.

This project gives you runtime building blocks for that job:

- Stable prompt layering: core system, tool schema, static context, session memory, history, runtime input
- Canonical serialization: sorted JSON keys, normalized whitespace, stripped timestamps and request IDs
- Metrics: cache hit rate, cached tokens, estimated saved cost with provider pricing presets
- Provider adapters: `CacheAwareOpenAI` (OpenAI / DeepSeek) and `CacheAwareClaude` (Anthropic)
- Session memory: local or LLM-powered summary and keyword extraction
- Benchmark tooling: naive vs optimized DeepSeek cache-hit comparisons

## Install

From source (recommended until PyPI release):

```bash
git clone https://github.com/xuzeTop1/llm-cache-optimizer.git
cd llm-cache-optimizer
pip install -e .
```

With optional provider dependencies:

```bash
pip install -e ".[openai]"      # OpenAI SDK (also works with DeepSeek)
pip install -e ".[anthropic]"   # Anthropic SDK for Claude
pip install -e ".[all]"         # Both providers
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

## Claude Adapter

Claude requires **explicit** `cache_control` breakpoints on stable content blocks (minimum 1024 tokens each). The `CacheAwareClaude` adapter handles this automatically:

```python
from llm_cache_optimizer import CacheAwareClaude

client = CacheAwareClaude(api_key="sk-ant-...")
client.add_core("You are a helpful coding assistant. ..." * 10)  # ≥1024 tokens
client.add_static_context("Project docs here...")

response = client.chat("Explain decorators.")
print(client.cache_report())
```

Under the hood, the adapter:
1. Merges all stable layers (core, tools, static context) into a single user message
2. Adds `cache_control: {"type": "ephemeral"}` if the block is large enough (≥ 4096 chars ≈ 1024 tokens)
3. Inserts an assistant turn ("Understood.") after the prefix — required by Claude for caching
4. Converts remaining history to Claude's Messages API format

Install:

```bash
pip install -e ".[anthropic]"
```

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

### Session Memory with LLM Summarizer

For higher-quality summaries, pass any callable that accepts history text and returns a summary. This can wrap an LLM call, a local model, or a custom business summarizer.

```python
from llm_cache_optimizer import CacheAwareClient, SessionMemory


def summarize_with_llm(history_text: str) -> str:
    """Return a higher-quality summary from your own LLM call."""

    return "User is building a cache-aware agent runtime with provider adapters."


client = CacheAwareClient(memory=SessionMemory(summarizer=summarize_with_llm))
client.chat("Build an OpenAI adapter and track cached token savings.")
memory = client.refresh_memory()
print(memory["summary"])
```

## Core API

```python
from llm_cache_optimizer import (
    CacheAwareClient,
    CacheAwareClaude,
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

### CacheMetrics

```python
from llm_cache_optimizer import CacheMetrics

metrics = CacheMetrics.from_provider("gpt-4o")
metrics.update_from_usage({
    "prompt_tokens": 1200,
    "prompt_tokens_details": {"cached_tokens": 900},
    "completion_tokens": 32,
})

print(metrics.report())
```

## Benchmark Results

The benchmark compares a naive agent that rebuilds its system prompt each turn against an optimized agent using `CacheAwareClient` with stable prefix layers.

Run it with DeepSeek:

```bash
pip install -e ".[openai]"
pip install -r benchmark/requirements.txt
set DEEPSEEK_API_KEY=sk-xxx
python benchmark/run_benchmark.py
```

Outputs:

- `benchmark/benchmark.csv`: per-turn cache hit rates and estimated costs
- `benchmark/benchmark.png`: chart comparing naive vs optimized cache hit curves

![Benchmark placeholder](benchmark/benchmark.png)

After running with a real API key, paste the generated summary table here:

| Metric | Naive | Optimized |
|---|---:|---:|
| Avg hit rate | TBD | TBD |
| Total cost | TBD | TBD |
| Savings | TBD | TBD |

## Examples

- [`examples/basic.py`](./examples/basic.py): minimal runtime example
- [`examples/deepseek_example.py`](./examples/deepseek_example.py): DeepSeek prefix-cache example
- [`examples/memory_demo.py`](./examples/memory_demo.py): local summary and keyword extraction
- [`examples/multi_provider_example.py`](./examples/multi_provider_example.py): five-layer provider-oriented agent loop
- [`examples/openai_compatible_example.py`](./examples/openai_compatible_example.py): Codex, OpenCode, and custom OpenAI-compatible gateways
- [`examples/claude_code_example.py`](./examples/claude_code_example.py): Claude Code pattern

## Current Roadmap

- v0.1.0: package structure, serializer, prompt layers, cache-aware client
- v0.2.0: OpenAI adapter, metrics, local session memory
- v0.3.0: Claude adapter, DeepSeek example, CI, custom summarizers, benchmark tooling
- v0.4.0: DeepSeek prefix diagnostics and provider-specific optimization reports
- Future: Gemini adapter, OpenCode hook, Claude Code skill, Codex middleware

## Cache-Aware Design Rules

- Put stable content first.
- Never rebuild the system prompt per turn.
- Keep tool schemas and static context in stable layers.
- Put runtime data, timestamps, retrieved chunks, and user input near the end.
- Normalize tool outputs before appending them to history.
- Compact long history into session memory instead of moving the prefix.

## Contributing

Contributions are welcome. Good first areas:

- Add provider adapters and usage-field parsers.
- Improve benchmark scenarios and publish reproducible results.
- Add cache diagnostics for DeepSeek, Claude, Gemini, and OpenAI-compatible gateways.
- Improve examples for Codex, Claude Code, OpenCode, and RAG agents.

Before opening a PR:

```bash
pip install -e ".[all]"
pytest tests/ -v
```

## License

MIT
