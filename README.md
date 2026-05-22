---

# LLM Prompt Cache Optimizer

A skill for maximizing prompt cache hit rates across any prefix-cache LLM (DeepSeek, Claude, Gemini, OpenAI), reducing token costs and latency.

---

## Core Mental Model

Most major LLMs cache on **prefix matching**: if the first N tokens of a new request exactly match a previously cached prefix, those tokens are served from cache at significantly reduced cost. The key invariant is simple:

```
✅ Turn 1:  [PREFIX] + [history_turn_1]
✅ Turn 2:  [PREFIX] + [history_turn_1] + [history_turn_2]   ← prefix hit
✅ Turn 3:  [PREFIX] + [history_turn_1] + [history_turn_2] + [history_turn_3]  ← hit

❌ Turn 2:  [PREFIX] + [history_turn_2]          ← reordered, miss
❌ Turn 2:  [new_PREFIX] + [history_turn_1]      ← mutated prefix, miss
❌ Turn 2:  [history_turn_1] + [PREFIX]          ← prefix moved, miss
```

**The golden rule: PREFIX is always first, always identical, never rebuilt.**

---

## Provider Comparison

| Provider             | Cache Mechanism                      | Min Cacheable | TTL                | Cached Token Cost |
| -------------------- | ------------------------------------ | ------------- | ------------------ | ----------------- |
| **DeepSeek**         | Automatic prefix cache               | 64 tokens     | ~Hours             | ~10% of normal    |
| **Anthropic Claude** | Explicit `cache_control` breakpoints | 1024 tokens   | 5 min (extendable) | ~10% of normal    |
| **Google Gemini**    | Explicit `cachedContent` API         | 32k tokens    | Configurable       | ~25% of normal    |
| **OpenAI**           | Automatic prefix cache               | 1128 tokens   | ~1 hour            | 50% of normal     |

---

## Audit Checklist

When reviewing Agent code, check each of these:

- [ ] System prompt is always the **first message**, never rebuilt per-turn
- [ ] Static context (RAG docs, tool schemas, persona) comes **immediately after** system prompt
- [ ] Neither is modified between turns (no f-string injection of turn-specific data)
- [ ] Conversation history is built by **appending**, not reconstructing
- [ ] Dynamic data (user query, date, retrieved chunks) goes at the **end**
- [ ] Tool results are appended in-place, not used to rebuild earlier messages
- [ ] (Claude only) `cache_control` breakpoints are on stable, large blocks only

---

## Canonical Pattern (all providers)

```python
# ── Build once at startup, never mutate ──────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant. ..."
STATIC_CONTEXT = "<docs>...your reusable RAG / tool schemas...</docs>"

PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_CONTEXT},
]

# ── Agent loop ────────────────────────────────────────────────────────────────
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

## Provider-Specific Notes

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
                "cache_control": {"type": "ephemeral"},  # mark cache boundary
            }
        ],
    },
    {"role": "assistant", "content": "Understood."},
]
# Check: response.usage.cache_read_input_tokens / cache_creation_input_tokens
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
# history appended automatically by the SDK
```

---

### OpenAI

Fully automatic for prompts ≥ 1128 tokens. No API changes needed.

```python
# Just keep prefix stable and ≥1128 tokens
# Check: response.usage.prompt_tokens_details.cached_tokens
```

---

## Anti-Patterns to Fix

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

## Hot Start Pattern

For scheduled jobs or long idle gaps between Agent runs:

```python
def warm_cache():
    """Reset cache TTL before the real workload starts."""
    client.chat.completions.create(
        model="...",
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
```

Call `warm_cache()` a few seconds before the first real task of each scheduled run.

---

## Diagnosing Cache Hit Rate

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

**Target:** >80% hit rate after turn 1 for Agents with large system prompts.

**If hit rate is low:**

1. Verify `PREFIX_MESSAGES` is the same object each call (`id()` check)
2. Look for f-strings or `.format()` anywhere in the prefix construction
3. Confirm total prefix exceeds the provider's minimum cacheable size
4. For Claude: confirm `cache_control` breakpoints are on blocks ≥1024 tokens

---

## Multi-Worker Pattern

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

## See Also

- `references/provider-cache-apis.md` — detailed per-provider API reference and edge cases
- `references/cache-metrics-logging.md` — structured logging helpers for tracking efficiency over time
