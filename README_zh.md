<p align="center">
  <b>🚀 停止为每次 API 调用支付全价。</b><br>
  跨 <b>DeepSeek、Claude、Gemini、OpenAI</b> 最大化前缀缓存命中率。<br>
  通过稳定的提示词结构，将令牌成本降低 <b>90%</b>。
</p>

---

# LLM 提示词缓存优化器

**系统性地最大化任何支持前缀缓存的 LLM 的缓存命中率**，通过保持提示词前缀在请求之间稳定不变，来降低令牌成本和延迟。

**主要优势：**
- ✅ 在重复查询中将 API 成本降低 **80-90%**
- ✅ 降低令牌处理延迟（缓存的令牌即时提供）
- ✅ 适用于所有主要供应商：DeepSeek、Claude、Gemini、OpenAI
- ✅ 生产就绪的模式和真实代码示例

---

## 📚 快速开始（30秒）

**黄金法则**：您的提示词前缀必须始终相同、始终在前、永不重建。

```python
# ✅ 在启动时构建一次
SYSTEM_PROMPT = "You are a helpful assistant..."
STATIC_CONTEXT = "<docs>...</docs>"
PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT},
]

# ✅ 在每一轮中重复使用
conversation_history = []

def chat(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    messages = PREFIX_MESSAGES + conversation_history  # 前缀始终在前
    response = client.chat.completions.create(model="...", messages=messages)
    return response.choices[0].message.content
```

**就这样**。这一个模式就能在后续每一轮都启用缓存命中。

---

## 🧠 核心心智模型

大多数主流 LLM 在**前缀匹配**上进行缓存：如果新请求的前 N 个令牌与之前缓存的前缀完全匹配，这些令牌将以显著降低的成本从缓存提供。

```
✅ 第 1 轮：  [前缀] + [历史第1轮]
✅ 第 2 轮：  [前缀] + [历史第1轮] + [历史第2轮]   ← 前缀命中
✅ 第 3 轮：  [前缀] + [历史第1轮] + [历史第2轮] + [历史第3轮]  ← 命中

❌ 第 2 轮：  [前缀] + [历史第2轮]          ← 重新排序，未命中
❌ 第 2 轮：  [新前缀] + [历史第1轮]      ← 前缀被修改，未命中
❌ 第 2 轮：  [历史第1轮] + [前缀]          ← 前缀被移动，未命中
```

**黄金法则：前缀始终在前、始终相同、永不重建。**

---

## 💰 供应商对比

| 供应商               | 缓存机制                      | 最小可缓存 | 有效期             | 缓存令牌成本 |
| -------------------- | ------------------------------------ | ------------- | ------------------ | ----------------- |
| **DeepSeek**         | 自动前缀缓存               | 64 令牌     | ~小时             | ~正常价格的 10%    |
| **Anthropic Claude** | 显式 `cache_control` 断点 | 1024 令牌   | 5 分钟（可延长） | ~正常价格的 10%    |
| **Google Gemini**    | 显式 `cachedContent` API         | 32k 令牌    | 可配置       | ~正常价格的 25%    |
| **OpenAI**           | 自动前缀缓存               | 1128 令牌   | ~1 小时            | 正常价格的 50%     |

---

## ✅ 审核清单

在审查 Agent 代码时使用：

- [ ] 系统提示是**第一条消息**，每一轮都不重建
- [ ] 静态上下文（RAG 文档、工具架构、角色设定）紧跟在系统提示后
- [ ] 两者在轮次之间都不被修改（没有针对特定轮次的 f 字符串注入）
- [ ] 对话历史通过**追加**来构建，而不是重建
- [ ] 动态数据（用户查询、日期、检索到的块）放在**末尾**
- [ ] 工具结果原地追加，不被用于重建之前的消息
- [ ] （仅 Claude）`cache_control` 断点只在稳定的大块上

---

## 🔧 规范模式（所有供应商）

```python
# ── 在启动时构建一次，永不修改 ──────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant. ..."
STATIC_CONTEXT = "<docs>...your reusable RAG / tool schemas...</docs>"

PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT},
]

# ── Agent 循环 ────────────────────────────────────────────────────────────
conversation_history = []   # 仅包含轮次特定的消息

def chat(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})

    # 稳定的前缀始终在前 — 这就是被缓存的部分
    messages = PREFIX_MESSAGES + conversation_history

    response = client.chat.completions.create(
        model="...",
        messages=messages,
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply
```

---

## 🏭 供应商特定实现

### DeepSeek

完全自动 — 无需更改 API，只需保持前缀稳定。

```python
client = OpenAI(api_key=..., base_url="https://api.deepseek.com/v1")
# 检查：response.usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens
```

---

### Anthropic Claude

需要在稳定块上使用**显式** `cache_control` 断点。最小块大小：**1024 令牌**。

```python
import anthropic
client = anthropic.Anthropic()

PREFIX_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT,
                "cache_control": {"type": "ephemeral"},  # 标记缓存边界
            }
        ],
    },
    {"role": "assistant", "content": "Understood."},
]
# 检查：response.usage.cache_read_input_tokens / cache_creation_input_tokens
```

---

### Google Gemini

使用单独的 `cachedContent` 对象；最小大小为 **32k 令牌**。

```python
import google.generativeai as genai
import datetime

cache = genai.caching.CachedContent.create(
    model="gemini-1.5-pro-001",
    system_instruction=SYSTEM_PROMPT,
    contents=[{"role": "user", "parts": [{"text": STATIC_CONTEXT}]}],
    ttl=datetime.timedelta(hours=1),
)
model = genai.GenerativeModel.from_cached_content(cache)
chat_session = model.start_chat()
# 历史记录由 SDK 自动追加
```

---

### OpenAI

对于 ≥1128 令牌的提示词完全自动。无需更改 API。

```python
# 只需保持前缀稳定且 ≥1128 令牌
# 检查：response.usage.prompt_tokens_details.cached_tokens
```

---

## ❌ 需要修复的反模式

### ❌ 动态数据注入到系统提示中

```python
# 错误 — 每一轮重建前缀，总是缓存未命中
messages = [{"role": "system", "content": f"Today: {datetime.now()}. Docs: {docs}"}]
```

**修复**：将 `datetime.now()` 和每轮 `docs` 移到最后的用户消息中。

---

### ❌ 从头开始重建历史

```python
# 错误 — 每一轮创建新的列表对象
messages = build_messages(all_past_messages)
```

**修复**：保持单个列表，只使用 `.append()`。

---

### ❌ 在历史中间插入上下文

```python
# 错误 — 破坏前缀连续性
messages = [system] + old_history + [retrieved_docs_msg] + new_history
```

**修复**：作为最后一条用户消息内容的一部分进行追加。

---

### ❌ 工具结果注入回系统提示

```python
# 错误 — 销毁缓存的前缀
messages = [{"role": "system", "content": SYSTEM + tool_output}] + history
```

**修复**：追加为 `{"role": "tool", "tool_call_id": ..., "content": ...}`。

---

## 🔥 高级模式

### 热启动（刷新缓存 TTL）

对于计划任务或 Agent 运行之间的长空闲间隙：

```python
def warm_cache():
    """在实际工作负载开始前重置缓存 TTL。"""
    client.chat.completions.create(
        model="...",
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
```

在每个计划运行的第一个真实任务前几秒钟调用 `warm_cache()`。

---

### 多工作线程模式

当并发运行多个工作线程时：

```python
# 模块级 — 由所有工作线程共享，缓存一次，由全部命中
PREFIX_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT + STATIC_CONTEXT}]

class AgentWorker:
    def __init__(self):
        self.history = []   # 工作线程本地

    def run(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})
        messages = PREFIX_MESSAGES + self.history   # 共享前缀 + 本地历史
        ...
```

---

## 📊 诊断缓存命中率

```python
def log_cache(usage, turn: int):
    # DeepSeek / OpenAI 兼容格式
    hit  = getattr(usage, "prompt_cache_hit_tokens", 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", 0)
    # OpenAI 嵌套格式
    if hasattr(usage, "prompt_tokens_details"):
        hit  = usage.prompt_tokens_details.cached_tokens or 0
        miss = usage.prompt_tokens - hit
    total = hit + miss
    rate  = hit / total if total else 0
    print(f"[第 {turn} 轮] cache={rate:.0%}  hit={hit}  miss={miss}")
```

**目标**：对于系统提示词较大的 Agent，第 1 轮之后命中率 >80%。

**如果命中率较低：**

1. 验证 `PREFIX_MESSAGES` 是否为同一对象（使用 `id()` 检查）
2. 在前缀构建中查找 f 字符串或 `.format()` 调用
3. 确认总前缀超过供应商的最小可缓存大小
4. 对于 Claude：确认 `cache_control` 断点只在 ≥1024 令牌的块上

---

## 📖 另请参阅

- `references/provider-cache-apis.md` — 详细的每个供应商 API 参考和边界情况
- `references/cache-metrics-logging.md` — 用于跟踪效率随时间变化的结构化日志记录助手
