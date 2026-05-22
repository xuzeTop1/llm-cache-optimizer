<p align="center">
  <b>🚀 Stop paying full price for every API call.</b><br>
  Maximize prefix-cache hit rates across <b>DeepSeek, Claude, Gemini, OpenAI</b>.<br>
  Cut token costs by up to <b>90%</b> with a stable prompt architecture.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Works%20with-Claude%20Code%20%7C%20Codex%20%7C%20OpenCode-blue" alt="Compatible">
  <img src="https://img.shields.io/badge/LLMs-DeepSeek%20%7C%20Claude%20%7C%20Gemini%20%7C%20OpenAI-green" alt="LLMs">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

---

# LLM Prompt Cache Optimizer

**Systematically maximize prompt cache hit rates across any prefix-cache LLM**, reducing token costs and latency by keeping your prompt prefix stable and unchanged between requests.

**Key benefits:**
- ✅ Cut API costs by **80–90%** on repeated queries
- ✅ Reduce token processing latency (cached tokens served instantly)
- ✅ Works with all major providers: DeepSeek, Claude, Gemini, OpenAI
- ✅ Compatible with Claude Code, OpenAI Codex CLI, OpenCode
- ✅ Production-ready patterns with real code examples

---

## 📚 Quick Start (30 seconds)

The **golden rule**: Your prompt prefix must always be the same, always first, never rebuilt.

```python
# ✅ Build once at startup
SYSTEM_PROMPT = "You are a helpful assistant..."
STATIC_CONTEXT = "<docs>...</docs>"
PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT},
]

# ✅ Reuse in every turn
conversation_history = []

def chat(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    messages = PREFIX_MESSAGES + conversation_history  # Prefix ALWAYS first
    response = client.chat.completions.create(model="...", messages=messages)
    return response.choices[0].message.content
```

**That's it.** This single pattern enables cache hits on every subsequent turn.

---

## 🧠 Core Mental Model

Most major LLMs cache on **prefix matching**: if the first N tokens of a new request exactly match a previously cached prefix, those tokens are served from cache at significantly reduced cost.

```
✅ Turn 1:  [PREFIX] + [history_turn_1]
✅ Turn 2:  [PREFIX] + [history_turn_1] + [history_turn_2]   ← prefix HIT
✅ Turn 3:  [PREFIX] + [history_turn_1] + [history_turn_2] + [history_turn_3]  ← HIT

❌ Turn 2:  [PREFIX] + [history_turn_2]          ← reordered, MISS
❌ Turn 2:  [new_PREFIX] + [history_turn_1]      ← mutated prefix, MISS
❌ Turn 2:  [history_turn_1] + [PREFIX]          ← prefix moved, MISS
```

---

## 💰 Provider Comparison

| Provider | Mechanism | Min Tokens | TTL | Cached Cost |
|---|---|---|---|---|
| **DeepSeek** | Automatic prefix cache | 64 | ~Hours | ~10% |
| **Anthropic Claude** | Explicit `cache_control` | 1,024 | 5 min (extendable) | ~10% |
| **Google Gemini** | Explicit `cachedContent` | 32,000 | Configurable | ~25% |
| **OpenAI** | Automatic prefix cache | 1,128 | ~1 hour | ~50% |

---

## 📦 Install to Your AI Coding Agent

### Claude Code
```bash
/plugin marketplace add anthropics/skills
# Or: cp SKILL.md ~/.claude/skills/llm-cache-optimizer.md
```

### OpenAI Codex CLI
```bash
mkdir -p ~/.agents/skills/llm-cache-optimizer
cp SKILL.md ~/.agents/skills/llm-cache-optimizer/
# Restart Codex or run /init
```

### OpenCode
```bash
mkdir -p ~/.opencode/skills/llm-cache-optimizer
cp SKILL.md ~/.opencode/skills/llm-cache-optimizer/
# In OpenCode: /init
```

> **AI agent users**: import `SKILL.md` directly. See [`examples/`](./examples/) for platform-specific agent loop patterns.

---

## 📂 What's Inside

| File | Purpose |
|---|---|
| `SKILL.md` | Full skill definition — import directly into your AI coding agent |
| [`examples/`](./examples/) | Platform-specific agent loops (Claude Code, Codex, OpenCode, multi-provider) |
| `README.md` | This page (GitHub landing) |

---

## 🔧 Canonical Pattern (all providers)

```python
# ── Build once at startup, never mutate ──────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant. ..."
STATIC_CONTEXT = "<docs>...your reusable RAG / tool schemas...</docs>"

PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT},
]

# ── Agent loop ───────────────────────────────────────────────────────────────
conversation_history = []   # turn-specific messages only

def chat(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})

    # Stable prefix ALWAYS first — this is what gets cached
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

## 🏭 Provider-Specific Implementation

### DeepSeek

Fully automatic — no API changes needed. Just keep the prefix stable.

```python
client = OpenAI(api_key=..., base_url="https://api.deepseek.com/v1")
# Check: response.usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens
```

---

### Anthropic Claude

Requires **explicit** `cache_control` breakpoints on stable blocks. Min block size: **1024 tokens**.

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
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {"role": "assistant", "content": "Understood."},
]
```

---

### Google Gemini

Uses a separate `cachedContent` object; min size is **32k tokens**.

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
```

---

### OpenAI

Fully automatic for prompts ≥ 1128 tokens. No API changes needed.

```python
# Just keep prefix stable and ≥1128 tokens
# Check: response.usage.prompt_tokens_details.cached_tokens
```

---

## ❌ Anti-Patterns to Fix

### ❌ Dynamic data injected into system prompt

```python
# BAD — rebuilds prefix every turn, always a cache miss
messages = [{"role": "system", "content": f"Today: {datetime.now()}. Docs: {docs}"}]
```

**Fix:** Move `datetime.now()` and per-turn `docs` into the last user message.

---

### ❌ Reconstructing history from scratch

```python
# BAD — creates new list object every turn
messages = build_messages(all_past_messages)
```

**Fix:** Keep a single list, only use `.append()`.

---

### ❌ Inserting context in the middle of history

```python
# BAD — breaks prefix continuity
messages = [system] + old_history + [retrieved_docs_msg] + new_history
```

**Fix:** Append retrieved docs as part of the last user message content.

---

### ❌ Tool result injected back into system prompt

```python
# BAD — destroys cached prefix
messages = [{"role": "system", "content": SYSTEM + tool_output}] + history
```

**Fix:** Append as `{"role": "tool", "tool_call_id": ..., "content": ...}`.

---

## 🔥 Advanced Patterns

### Hot Start (Refresh Cache TTL)

```python
def warm_cache():
    """Reset cache TTL before the real workload starts."""
    client.chat.completions.create(
        model="...",
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
```

### Multi-Worker Pattern

```python
# Module-level — shared by all workers, cached once, hit by all
PREFIX_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT + STATIC_CONTEXT}]

class AgentWorker:
    def __init__(self):
        self.history = []   # worker-local

    def run(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})
        messages = PREFIX_MESSAGES + self.history   # shared prefix + local history
        ...
```

---

## ✅ Audit Checklist

- [ ] System prompt is always the **first message**, never rebuilt per-turn
- [ ] Static context (RAG docs, tool schemas, persona) comes **immediately after**
- [ ] Neither is modified between turns (no f-string injection of turn-specific data)
- [ ] Conversation history is built by **appending**, not reconstructing
- [ ] Dynamic data (user query, date, retrieved chunks) goes at the **end**
- [ ] Tool results are appended in-place, not used to rebuild earlier messages
- [ ] (Claude only) `cache_control` breakpoints are on stable, large blocks only

---

## 📊 Diagnosing Cache Hit Rate

```python
def log_cache(usage, turn: int):
    # DeepSeek / OpenAI-compatible
    hit  = getattr(usage, "prompt_cache_hit_tokens", 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", 0)
    # OpenAI nested format
    if hasattr(usage, "prompt_tokens_details"):
        hit  = usage.prompt_tokens_details.cached_tokens or 0
        miss = usage.prompt_tokens - hit
    total = hit + miss
    rate  = hit / total if total else 0
    print(f"[Turn {turn}] cache={rate:.0%}  hit={hit}  miss={miss}")
```

**Target:** >80% hit rate after turn 1.

---

## 📖 See Also

- [`examples/`](./examples/) — Platform-specific agent loop examples
- `references/provider-cache-apis.md` — per-provider API details and edge cases
- `references/cache-metrics-logging.md` — structured logging helpers

## 📄 License

MIT © 2025
