# LLM Cache Optimizer

面向 Agent 的 Prompt Cache Runtime。

`llm-cache-optimizer` 帮助开发者构建更容易命中前缀缓存的 LLM Agent：稳定 prompt 前缀、规范化易变输入、统计缓存收益，并把长对话压缩成可复用的 session memory。

当前已经支持 OpenAI 风格的 chat messages，并提供 OpenAI SDK 适配器。核心模块保持 provider-neutral，后续可以继续扩展到 DeepSeek、Claude、Gemini、OpenAI-compatible API、Codex、Claude Code 和 OpenCode 场景。

## 为什么需要它

很多 LLM 服务商都支持前缀缓存。只要新请求开头的一段 prompt 与之前处理过的 prompt prefix 匹配，服务商就可以复用缓存 token，从而降低延迟和成本。

真正困难的地方不是“知道有缓存”，而是在真实 Agent 多轮工作流里保持 prompt prefix 稳定。

这个项目提供以下运行时能力：

- 稳定的 prompt 分层：core system、tool schema、static context、session memory、history、runtime input
- 规范化序列化：JSON key 排序、whitespace 归一化、移除时间戳和 request id
- 指标统计：缓存命中率、cached tokens、预估节省成本
- Provider adapter：`CacheAwareOpenAI`
- Session memory：本地摘要和关键词提取，用于更 cache-friendly 的历史压缩

## 安装

```bash
pip install llm-cache-optimizer
```

如果要使用 OpenAI adapter：

```bash
pip install "llm-cache-optimizer[openai]"
```

## 快速开始

```python
from llm_cache_optimizer import CacheAwareClient

client = CacheAwareClient()
client.add_core("You are a concise coding assistant.")
client.add_tool_schema({"name": "read_file", "description": "Read a workspace file."})
client.add_static_context("Project: llm-cache-optimizer")

messages = client.chat("Show me the cache-aware message layout.")
print(messages)
```

生成消息时，稳定内容会始终排在动态输入前面：

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

它包装的是：

```python
OpenAI().chat.completions.create(...)
```

并会自动构建 cache-ordered messages，同时读取常见 usage 字段：

- `usage.prompt_tokens`
- `usage.prompt_tokens_details.cached_tokens`
- `usage.completion_tokens`

## DeepSeek

DeepSeek 很适合作为缓存 benchmark 的低成本 provider：它支持自动前缀缓存，最小可缓存前缀大约 64 tokens，cached input token 成本大约是普通 input token 的 10%。

使用 OpenAI-compatible adapter，并把 `base_url` 指向 DeepSeek：

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

DeepSeek-compatible usage 对象通常会暴露缓存字段，例如：

- `usage.prompt_cache_hit_tokens`
- `usage.prompt_cache_miss_tokens`

`CacheMetrics.update_from_usage()` 会在这些字段存在时读取 `prompt_cache_hit_tokens`。

## Session Memory

长时间运行的 Agent 不应该不断重建不稳定的 prompt prefix。可以使用 session memory 对历史进行摘要，并提取可复用关键词。

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

示例输出：

```python
{
    "summary": "user: We are turning this repo into a Python runtime. user: Next we need an OpenAI adapter and a memory demo.",
    "keywords": ["adapter", "memory", "openai", "runtime"],
}
```

## 核心 API

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

## 示例

- [`examples/basic.py`](./examples/basic.py)：最小 runtime 示例
- [`examples/memory_demo.py`](./examples/memory_demo.py)：本地摘要与关键词提取示例
- [`examples/multi_provider_example.py`](./examples/multi_provider_example.py)：多 provider Agent loop 模式
- [`examples/opencode_example.py`](./examples/opencode_example.py)：OpenCode 模式
- [`examples/claude_code_example.py`](./examples/claude_code_example.py)：Claude Code 模式
- [`examples/codex_example.py`](./examples/codex_example.py)：Codex 模式

## 当前路线图

- v0.1.0：package 结构、serializer、prompt layers、cache-aware client
- v0.2.0：OpenAI adapter、metrics、本地 session memory
- v0.3.0：benchmark system，对比 baseline 与 optimized agent loop
- v0.4.0：DeepSeek prefix diagnostics 和 provider-specific optimization report
- 后续：Claude cache-control adapter、OpenCode hook、Claude Code skill、Codex middleware

## Cache-Aware 设计规则

- 稳定内容放在最前面。
- 不要每一轮都重建 system prompt。
- tool schemas 和 static context 放在稳定层。
- runtime data、时间戳、检索片段、用户输入尽量放在末尾。
- tool output 进入 history 前先规范化。
- 长历史压缩成 session memory，不要移动 prefix。

## License

MIT
