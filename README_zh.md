# LLM Cache Optimizer

![Tests](https://github.com/xuzeTop1/llm-cache-optimizer/actions/workflows/test.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/llm-cache-optimizer.svg)](https://pypi.org/project/llm-cache-optimizer/)

面向 Agent 的 Prompt Cache Runtime。

`llm-cache-optimizer` 帮助开发者构建更容易命中前缀缓存的 LLM Agent：稳定 prompt 前缀、规范化易变输入、统计 cached token 收益，并把长对话压缩成可复用的 session memory。

当前支持 OpenAI 风格的 chat messages，并提供 OpenAI / DeepSeek adapter 和 Anthropic Claude adapter。核心模块保持 provider-neutral，后续可以继续扩展到 Gemini、Codex、Claude Code 和 OpenCode 场景。

## 为什么需要它

很多 LLM 服务商都支持前缀缓存。只要新请求开头的一段 prompt 与之前处理过的 prompt prefix 匹配，服务商就可以复用缓存 token，从而降低延迟和成本。

真正困难的地方不是"知道有缓存"，而是在真实 Agent 多轮工作流里保持 prompt prefix 稳定。

这个项目提供以下运行时能力：

- 稳定的 prompt 分层：core system、tool schema、static context、session memory、history、runtime input
- 规范化序列化：JSON key 排序、whitespace 归一化、移除时间戳和 request id
- 指标统计：缓存命中率、cached tokens、基于 provider 价格预设的预估节省成本
- Provider adapters：`CacheAwareOpenAI`（OpenAI / DeepSeek）和 `CacheAwareClaude`（Anthropic）
- Session memory：本地摘要或 LLM 摘要，以及关键词提取
- Benchmark 工具：对比 naive 与 optimized DeepSeek cache hit 曲线

## 安装

从 PyPI 安装（推荐）：

```bash
pip install llm-cache-optimizer
```

使用可选的 provider 依赖：

```bash
pip install "llm-cache-optimizer[openai]"      # OpenAI SDK（也适用于 DeepSeek）
pip install "llm-cache-optimizer[anthropic]"   # Anthropic SDK（Claude）
pip install "llm-cache-optimizer[all]"         # 所有 provider + 开发工具
```

从源码安装（开发）：

```bash
git clone https://github.com/xuzeTop1/llm-cache-optimizer.git
cd llm-cache-optimizer
pip install -e ".[all,dev]"
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

## ChatGPT / OpenAI Adapter

OpenAI API 上的 ChatGPT 模型会对符合条件的长 prompt 自动进行前缀缓存。你不需要添加 provider 专用的 cache marker；最关键的是让每次请求开头的稳定内容保持完全一致。OpenAI 会通过 `usage.prompt_tokens_details.cached_tokens` 返回实际命中的缓存 token 数。

```python
from llm_cache_optimizer import CacheAwareOpenAI

client = CacheAwareOpenAI(api_key="...", model="gpt-4o-mini")
client.add_core(
    "You are ChatGPT, a concise coding assistant. "
    "Keep these instructions stable across the whole session."
)
client.add_tool_schema({
    "name": "read_file",
    "description": "Read a file from the current project.",
})
client.add_static_context("Stable project docs, repository notes, and coding rules go here.")

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

在 ChatGPT 类应用里，建议把长期稳定的人设、系统指令、工具 schema、项目上下文和 examples 放进 `add_core()`、`add_tool_schema()` 和 `add_static_context()`。用户问题、时间戳、检索片段和每轮变化的状态放在 `chat()` 或 history 末尾，避免破坏前缀缓存。

## Claude Adapter

Claude 需要**显式**的 `cache_control` 标记来指定哪些内容块应该被缓存（每个块最少 1024 tokens）。`CacheAwareClaude` adapter 会自动处理这些：

```python
from llm_cache_optimizer import CacheAwareClaude

client = CacheAwareClaude(api_key="sk-ant-...")
client.add_core("You are a helpful coding assistant. ..." * 10)  # ≥1024 tokens
client.add_static_context("Project docs here...")

response = client.chat("Explain decorators.")
print(client.cache_report())
```

底层实现：
1. 将开头的稳定层（core、tools、static context）合并到 Claude 原生的 `system` 字段
2. 如果 system block 足够大（≥ 4096 字符 ≈ 1024 tokens），自动添加 `cache_control: {"type": "ephemeral"}`
3. 将剩余 user、assistant 和 tool history 转换为 Claude Messages API 格式
4. 读取 Anthropic usage 字段，例如 `cache_read_input_tokens`

安装：

```bash
pip install -e ".[anthropic]"
```

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

### 使用 LLM Summarizer 的 Session Memory

如果想要更高质量的摘要，可以传入任意 callable。它接收 history text，并返回 summary。这个 callable 可以包装一次 LLM 调用、本地模型，或者你自己的业务摘要器。

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

## 核心 API

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

benchmark 会对比两种 Agent：

- Naive：每一轮都重建 system prompt，并注入 timestamp
- Optimized：使用 `CacheAwareClient`，保持稳定 prefix layers

使用 DeepSeek 运行：

```bash
pip install -e ".[openai]"
pip install -r benchmark/requirements.txt
set DEEPSEEK_API_KEY=sk-xxx
python benchmark/run_benchmark.py
```

输出：

- `benchmark/benchmark.csv`：每轮 cache hit rate 和预估成本
- `benchmark/benchmark.png`：naive vs optimized cache hit 曲线图

![Benchmark placeholder](benchmark/benchmark.png)

用真实 API key 跑完后，可以把脚本打印的 summary table 更新到这里：

| Metric | Naive | Optimized |
|---|---:|---:|
| Avg hit rate | TBD | TBD |
| Total cost | TBD | TBD |
| Savings | TBD | TBD |

## 示例

- [`examples/basic.py`](./examples/basic.py)：最小 runtime 示例
- [`examples/deepseek_example.py`](./examples/deepseek_example.py)：DeepSeek prefix-cache 示例
- [`examples/memory_demo.py`](./examples/memory_demo.py)：本地摘要与关键词提取示例
- [`examples/multi_provider_example.py`](./examples/multi_provider_example.py)：五层结构的多 provider Agent loop
- [`examples/openai_compatible_example.py`](./examples/openai_compatible_example.py)：Codex、OpenCode 和自定义 OpenAI-compatible gateway
- [`examples/claude_code_example.py`](./examples/claude_code_example.py)：Claude Code 模式

## 当前路线图

- v0.1.0：package 结构、serializer、prompt layers、cache-aware client
- v0.2.0：OpenAI adapter、metrics、本地 session memory
- v0.3.0：Claude adapter、DeepSeek 示例、CI、自定义 summarizer、benchmark 工具
- v0.4.0：DeepSeek prefix diagnostics 和 provider-specific optimization report
- 后续：Gemini adapter、OpenCode hook、Claude Code skill、Codex middleware

## Cache-Aware 设计规则

- 稳定内容放在最前面。
- 不要每一轮都重建 system prompt。
- tool schemas 和 static context 放在稳定层。
- runtime data、时间戳、检索片段、用户输入尽量放在末尾。
- tool output 进入 history 前先规范化。
- 长历史压缩成 session memory，不要移动 prefix。

## Contributing

欢迎贡献。适合优先做的方向：

- 新增 provider adapters 和 usage 字段解析。
- 改进 benchmark 场景，并发布可复现结果。
- 为 DeepSeek、Claude、Gemini 和 OpenAI-compatible gateway 增加 cache diagnostics。
- 改进 Codex、Claude Code、OpenCode 和 RAG Agent 示例。

提交 PR 前建议运行：

```bash
pip install -e ".[all]"
pytest tests/ -v
```

## License

MIT
