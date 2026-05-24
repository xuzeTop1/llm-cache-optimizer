# LLM Cache Optimizer — Claude Code Skill

Use this skill when building, reviewing, or debugging LLM agents that use prefix-based prompt caching (DeepSeek, Claude, Gemini, OpenAI).

## Core Rules

When writing or reviewing agent code, enforce these patterns:

### 1. Prefix Stability (CRITICAL)

The system prompt and static context must be **identical across every turn**. Never rebuild them.

```python
# CORRECT: build once, reuse forever
PREFIX = [
    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STATIC_DOCS},
]

def chat(user_input):
    messages = PREFIX + history  # prefix always first
    ...
```

```python
# WRONG: rebuilds prefix every turn
def chat(user_input):
    messages = [{"role": "system", "content": f"Time: {now()}\nDocs: {docs}"}]
    ...
```

### 2. Six-Layer Architecture

Structure prompts in this order (stable → dynamic):

```
Layer 1: core_system     — persona, constraints, output format     (NEVER changes)
Layer 2: tool_schema     — function signatures, tool descriptions  (changes on deploy)
Layer 3: static_context  — RAG docs, knowledge base                (changes per task)
Layer 4: session_memory  — user prefs, past summaries              (changes per session)
Layer 5: history         — conversation turns                       (append-only)
Layer 6: runtime         — current user input, dynamic data         (changes per turn)
```

### 3. Canonicalize Tool Outputs

Before appending tool results to history, strip volatile fields:

```python
import re, json

def canonicalize(result):
    if isinstance(result, str):
        result = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^"]*', '<ts>', result)
        return result
    if isinstance(result, dict):
        VOLATILE = {"timestamp", "ts", "request_id", "trace_id", "created_at", "updated_at"}
        cleaned = {k: v for k, v in result.items() if k not in VOLATILE}
        return json.dumps(cleaned, sort_keys=True)
    return json.dumps(result, sort_keys=True)
```

### 4. Append-Only History

Never reconstruct history. Only use `.append()`.

```python
# CORRECT
history.append({"role": "user", "content": user_input})
history.append({"role": "assistant", "content": response})

# WRONG
history = build_messages(all_past_messages)  # creates new list every turn
```

### 5. Context Compaction

When history exceeds the context window, compact the **old portion only** — never touch the prefix.

```python
def compact(history, prefix):
    if len(history) < 40:  # 20 turns
        return history
    old = history[:20]
    recent = history[20:]
    summary = llm_summarize(prefix + old)
    return [{"role": "user", "content": f"<summary>{summary}</summary>"}] + recent
```

## Provider-Specific Notes

### DeepSeek / OpenAI (automatic caching)
- No code changes needed — just keep prefix stable
- Minimum: ~64 tokens (DeepSeek) / ~1128 tokens (OpenAI)
- Check: `usage.prompt_cache_hit_tokens` (DeepSeek) or `usage.prompt_tokens_details.cached_tokens` (OpenAI)

### Claude (explicit cache_control)
- Place `cache_control: {"type": "ephemeral"}` on stable blocks
- Each block must be ≥1024 tokens
- Insert an assistant turn after the cached user content

### Gemini (explicit cachedContent)
- Create a `CachedContent` object for stable prefix
- Minimum: 32k tokens
- Reuse across turns via `GenerativeModel.from_cached_content()`

## Audit Checklist

When reviewing agent code, verify:

- [ ] System prompt is always the **first message**, never rebuilt per-turn
- [ ] No `datetime.now()`, `uuid4()`, or f-strings in prefix construction
- [ ] History is **append-only** — never reconstructed from scratch
- [ ] Tool results are **canonicalized** before appending
- [ ] Context compaction operates **only on dynamic history**, never on prefix
- [ ] Multi-worker systems share prefix via registry, not independent construction
- [ ] Cache hit rate is logged after each turn

## Anti-Patterns to Flag

| Pattern | Problem | Fix |
|---------|---------|-----|
| `f"Time: {datetime.now()}"` in system prompt | Breaks prefix every turn | Move to last user message |
| `messages = build_all_messages()` | Reconstructs history | Use `history.append()` |
| `[system] + old + [new_docs] + new` | Inserts in middle | Append docs to last user message |
| Tool output with timestamps in history | Breaks cache on identical data | Canonicalize first |

## Quick Start Template

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="...")

PREFIX = [{"role": "system", "content": "You are a helpful assistant. ..."}]
history = []

def chat(user_input):
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="...",
        messages=PREFIX + history,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    
    # Log cache metrics
    hit = getattr(response.usage, 'prompt_cache_hit_tokens', 0)
    total = response.usage.prompt_tokens
    print(f"Cache: {hit}/{total} = {hit/total:.0%}")
    
    return reply
```

## Install

Copy this file to your Claude Code skills directory:

```bash
cp docs/claude-code/cache-optimizer.md ~/.claude/skills/cache-optimizer.md
```

Or use the Python library directly:

```bash
pip install -e ".[all]"
```
