---
name: llm-cache-optimizer
description: >
  Use when building, reviewing, or debugging LLM agents with prefix-based prompt caching (DeepSeek, Claude, Gemini, OpenAI). Triggers: prompt caching, cache hit rate, token cost, agent loop, multi-turn conversation, system prompt, context window, tool results. Covers provider-specific APIs, prompt layering, context compaction, multi-worker prefix registry, tool result canonicalization, and cache warm-up. Compatible with Claude Code, OpenAI Codex CLI, OpenCode, and all prefix-cache LLM providers.
---

# LLM Prompt Cache Optimizer

Production-grade skill for maximizing prompt cache hit rates across any prefix-cache LLM (DeepSeek, Claude, Gemini, OpenAI), covering architecture design, multi-agent systems, and context lifecycle management.

Compatible with: **Claude Code** · **OpenAI Codex CLI** · **OpenCode** · any agent framework using LLM APIs.

---

## Core Mental Model

All major LLMs use **prefix matching** for caching: if the first N tokens of a new request exactly match a cached prefix, those tokens are served at significantly reduced cost.

```
✅ Turn 1:  [PREFIX] + [history_1]
✅ Turn 2:  [PREFIX] + [history_1] + [history_2]        ← prefix hit
✅ Turn 3:  [PREFIX] + [history_1] + [history_2] + [history_3]  ← hit

❌  [PREFIX] + [history_2]           ← reordered, miss
❌  [new_PREFIX] + [history_1]       ← mutated prefix, miss
❌  [history_1] + [PREFIX]           ← prefix moved, miss
```

**The golden rule: PREFIX is always first, always identical, never rebuilt.**

---

## Provider Comparison

| Provider | Mechanism | Min Cacheable | TTL | Cached Cost |
|---|---|---|---|---|
| **DeepSeek** | Automatic prefix cache | 64 tokens | ~Hours | ~10% |
| **Anthropic Claude** | Explicit `cache_control` | 1024 tokens | 5 min (extendable) | ~10% |
| **Google Gemini** | Explicit `cachedContent` | 32k tokens | Configurable | ~25% |
| **OpenAI** | Automatic prefix cache | 1128 tokens | ~1 hour | ~50% |

---

## Prompt Layering Architecture

Production agents should not use a single flat PREFIX. Instead, use **layered prefixes** with different stability and TTL characteristics:

```
┌─────────────────────────────────────────┐
│  LAYER 1: CORE_SYSTEM                   │  ← Never changes. Cached indefinitely.
│  (persona, constraints, output format)  │    ~hundreds of tokens
├─────────────────────────────────────────┤
│  LAYER 2: TOOLS                         │  ← Changes only on deploy.
│  (tool schemas, function signatures)    │    ~thousands of tokens
├─────────────────────────────────────────┤
│  LAYER 3: RAG_STATIC                    │  ← Changes per task/session start.
│  (knowledge base, reference docs)       │    ~tens of thousands of tokens
├─────────────────────────────────────────┤
│  LAYER 4: SESSION_MEMORY                │  ← Changes per session.
│  (user prefs, past summaries)           │    ~hundreds of tokens
├─────────────────────────────────────────┤
│  LAYER 5: DYNAMIC_HISTORY               │  ← Append-only, per turn.
│  (conversation turns, tool results)     │    grows over time
└─────────────────────────────────────────┘
```

**Key principle:** Layers 1-3 form the "compilable prefix" — treat them like compiled bytecode. They should never be rebuilt at runtime.

```python
# Build each layer once; combine in strict order
LAYER_CORE   = "You are a helpful assistant. Always respond in JSON. ..."
LAYER_TOOLS  = load_tool_schemas()          # loaded once at startup
LAYER_RAG    = load_knowledge_base()        # loaded once per task type

# The prefix = stable layers concatenated in order
PREFIX_MESSAGES = [
    {"role": "system", "content": "\n\n".join([LAYER_CORE, LAYER_TOOLS, LAYER_RAG])},
]

# Layers 4-5: runtime only
session_memory  = []   # summaries, user prefs — reset per session
dynamic_history = []   # append-only conversation turns
```

For **Claude**, place `cache_control` breakpoints at the boundary of each stable layer:

```python
PREFIX_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": LAYER_CORE,  "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": LAYER_TOOLS, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": LAYER_RAG,   "cache_control": {"type": "ephemeral"}},
        ],
    },
    {"role": "assistant", "content": "Understood."},
]
```

---

## Prefix Hash Registry (Multi-Worker)

In multi-agent or parallel worker systems, all workers sharing the same prefix should reference the **same registry entry** — not independently construct identical prefixes.

```python
import hashlib, json
from typing import Optional

class PrefixRegistry:
    """
    Central registry for shared prefixes.
    Workers look up by hash to confirm they're using the same cached prefix.
    """
    _store: dict[str, list] = {}

    @classmethod
    def register(cls, messages: list) -> str:
        key = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        cls._store[key] = messages
        return key

    @classmethod
    def get(cls, key: str) -> Optional[list]:
        return cls._store.get(key)


# At startup — register once
PREFIX_KEY = PrefixRegistry.register(PREFIX_MESSAGES)

class AgentWorker:
    def __init__(self, prefix_key: str):
        self.prefix = PrefixRegistry.get(prefix_key)   # shared reference
        self.history = []                               # worker-local

    def run(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})
        messages = self.prefix + self.history           # prefix always first
        response = client.chat.completions.create(model="...", messages=messages)
        ...
```

> In distributed systems, serialize the prefix hash into your task payload so downstream workers can verify they're using the same prefix version before making API calls.

---

## Context Window Compaction

Append-only history will eventually exceed the context window. A production agent needs a compaction strategy.

### Strategy 1: Rolling Window (simple)

```python
MAX_HISTORY_TURNS = 20

def get_messages(prefix, history, user_input):
    history.append({"role": "user", "content": user_input})
    # Keep only the last N turns; prefix is always preserved
    trimmed = history[-MAX_HISTORY_TURNS * 2:]
    return prefix + trimmed
```

### Strategy 2: Summary Buffer (recommended)

```python
SUMMARY_THRESHOLD = 30   # turns before compaction triggers

def maybe_compact(history: list, prefix: list) -> list:
    if len(history) < SUMMARY_THRESHOLD * 2:
        return history

    # Summarize the oldest half of history
    old_turns = history[:SUMMARY_THRESHOLD]
    recent    = history[SUMMARY_THRESHOLD:]

    summary_response = client.chat.completions.create(
        model="...",
        messages=prefix + old_turns + [{
            "role": "user",
            "content": "Summarize the conversation so far in 3-5 bullet points, "
                       "preserving all key decisions and facts."
        }],
        max_tokens=300,
    )
    summary_text = summary_response.choices[0].message.content

    # Replace old turns with a compact summary message
    summary_msg = {
        "role": "user",
        "content": f"<conversation_summary>\n{summary_text}\n</conversation_summary>"
    }
    return [summary_msg] + recent

# In the agent loop
dynamic_history = maybe_compact(dynamic_history, PREFIX_MESSAGES)
```

### Strategy 3: Semantic Checkpoint (long-running agents)

```python
import json, time

def save_checkpoint(history: list, path: str):
    """Persist compacted history to disk for resumable agents."""
    with open(path, "w") as f:
        json.dump({"ts": time.time(), "history": history}, f)

def load_checkpoint(path: str) -> list:
    with open(path) as f:
        return json.load(f)["history"]
```

> **Rule:** Compaction always operates on `dynamic_history` only — **never** touch the PREFIX layers.

---

## Tool Result Canonicalization

Unstable tool outputs (timestamps, UUIDs, floating-point noise) break cache continuity even when the underlying data is the same. Canonicalize before appending to history.

```python
import json, re
from typing import Any

def canonicalize_tool_result(result: Any) -> str:
    """
    Normalize tool output to maximize cache stability:
    - Remove timestamps and volatile fields
    - Sort JSON keys deterministically
    - Truncate high-precision floats
    - Strip internal IDs that change per-call
    """
    if isinstance(result, str):
        # Remove ISO timestamps: 2026-05-22T11:32:22.123Z
        result = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?',
            '<timestamp>',
            result
        )
        return result

    if isinstance(result, dict):
        # Drop known volatile keys
        VOLATILE_KEYS = {"timestamp", "ts", "request_id", "trace_id", "latency_ms",
                         "created_at", "updated_at", "expires_at", "_debug"}
        cleaned = {k: v for k, v in result.items() if k not in VOLATILE_KEYS}

        # Truncate floats to 4 decimal places
        def _clean_value(v):
            if isinstance(v, float):
                return round(v, 4)
            if isinstance(v, dict):
                return {k2: _clean_value(v2) for k2, v2 in v.items()}
            if isinstance(v, list):
                return [_clean_value(i) for i in v]
            return v

        cleaned = {k: _clean_value(v) for k, v in cleaned.items()}
        return json.dumps(cleaned, sort_keys=True, ensure_ascii=False)

    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def append_tool_result(history: list, tool_call_id: str, raw_result: Any):
    """Always canonicalize before appending to preserve cache."""
    history.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": canonicalize_tool_result(raw_result),
    })
```

---

## Canonical Agent Loop (full production pattern)

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["API_KEY"], base_url="...")

# ── Layer construction (once at startup) ─────────────────────────────────────
PREFIX_MESSAGES = [
    {"role": "system", "content": "\n\n".join([LAYER_CORE, LAYER_TOOLS, LAYER_RAG])},
]
PREFIX_KEY = PrefixRegistry.register(PREFIX_MESSAGES)

# ── Agent loop ────────────────────────────────────────────────────────────────
dynamic_history: list = []

def chat(user_input: str) -> str:
    global dynamic_history

    # 1. Compact if needed (never touches PREFIX)
    dynamic_history = maybe_compact(dynamic_history, PREFIX_MESSAGES)

    # 2. Append user turn
    dynamic_history.append({"role": "user", "content": user_input})

    # 3. PREFIX always first
    messages = PREFIX_MESSAGES + dynamic_history

    response = client.chat.completions.create(model="...", messages=messages)
    msg = response.choices[0].message

    # 4. Handle tool calls with canonicalization
    if msg.tool_calls:
        dynamic_history.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            raw = execute_tool(tc.function.name, tc.function.arguments)
            append_tool_result(dynamic_history, tc.id, raw)   # canonicalized
    else:
        dynamic_history.append({"role": "assistant", "content": msg.content})

    # 5. Log cache metrics
    log_cache(response.usage, len(dynamic_history))

    return msg.content
```

---

## Platform-Specific Notes

### Claude Code
Claude Code uses `cache_control` with `ephemeral` type. Place breakpoints at each stable layer boundary, ensuring each marked block is ≥1024 tokens:
```python
{"type": "text", "text": PREFIX_CONTENT, "cache_control": {"type": "ephemeral"}}
```

### OpenAI Codex CLI
Codex uses automatic prefix caching. No API changes needed — just ensure prefix stability and size ≥1128 tokens. Use the OpenAI SDK standard:
```python
response.usage.prompt_tokens_details.cached_tokens  # check hit
```

### OpenCode
OpenCode proxies through your configured model. Follow the provider's native caching mechanism. For Claude-backed: use `cache_control`. For OpenAI-backed: automatic.

---

## Audit Checklist

When reviewing Agent code, check each of these:

- [ ] System prompt is always the **first message**, never rebuilt per-turn
- [ ] Static layers (core, tools, RAG) come **immediately after**, in stable order
- [ ] No f-strings, `datetime.now()`, or runtime variables in any PREFIX layer
- [ ] Conversation history is **append-only** — never reconstructed
- [ ] Dynamic data goes at the **end** of messages, never inserted in the middle
- [ ] Tool results are **canonicalized** before appending (no timestamps, no volatile fields)
- [ ] Context compaction operates **only on dynamic history**, never on PREFIX
- [ ] Multi-worker systems share PREFIX via registry, not by independent construction
- [ ] (Claude) `cache_control` breakpoints placed at each stable layer boundary, blocks ≥1024 tokens
- [ ] (Gemini) `cachedContent` created once per task type, not per request

---

## Hot Start Pattern

For scheduled jobs or long idle gaps:

```python
def warm_cache():
    """Reset cache TTL before the real workload starts."""
    client.chat.completions.create(
        model="...",
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
```

---

## Cache Hit Rate Diagnostics

```python
def log_cache(usage, turn: int):
    hit  = getattr(usage, "prompt_cache_hit_tokens", 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", 0)
    if hasattr(usage, "prompt_tokens_details"):       # OpenAI format
        hit  = usage.prompt_tokens_details.cached_tokens or 0
        miss = usage.prompt_tokens - hit
    total = hit + miss
    rate  = hit / total if total else 0
    print(f"[Turn {turn}] cache={rate:.0%}  hit={hit}  miss={miss}")
```

**Target:** >80% hit rate after turn 1.

**If hit rate is low:**

1. `id(PREFIX_MESSAGES)` — verify same object across calls
2. Search for f-strings or `.format()` anywhere in prefix construction
3. Check total prefix size vs. provider minimum (see table above)
4. Run `canonicalize_tool_result` on a sample output and check for volatile fields
5. Confirm no compaction logic is touching PREFIX layers

---

## See Also

- `references/provider-cache-apis.md` — per-provider API details and edge cases
- `references/cache-metrics-logging.md` — structured logging helpers
