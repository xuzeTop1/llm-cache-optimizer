# Cache-Aware Agent Instructions

These instructions help AI coding agents (Codex, Claude Code, OpenCode) generate LLM agent code that maximizes prefix cache hit rates.

## Prefix Caching Rules

When generating or reviewing LLM agent code, follow these rules:

### Rule 1: Stable Prefix First

System prompt and static context must be **built once** and placed at the **beginning** of every request. Never rebuild them per turn.

### Rule 2: Six-Layer Ordering

```
1. core_system     — persona, rules, output format        (immutable)
2. tool_schema     — function definitions                  (deploy-time)
3. static_context  — RAG docs, knowledge base              (task-time)
4. session_memory  — user prefs, summaries                 (session-time)
5. history         — conversation turns                    (append-only)
6. runtime         — current input, dynamic data           (per-turn)
```

### Rule 3: Append-Only History

Never reconstruct the message list. Use `history.append()` only.

### Rule 4: Canonicalize Tool Outputs

Strip timestamps, UUIDs, and volatile fields before appending tool results to history:

```python
import re, json

def canonicalize(result):
    if isinstance(result, str):
        return re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^"]*', '<ts>', result)
    if isinstance(result, dict):
        clean = {k: v for k, v in result.items()
                 if k not in {"timestamp", "ts", "request_id", "created_at"}}
        return json.dumps(clean, sort_keys=True)
    return json.dumps(result, sort_keys=True)
```

### Rule 5: Compact Only Dynamic History

When the context window is full, summarize old history but **never touch the prefix**.

## Provider Cheat Sheet

| Provider | Caching | Min Tokens | Check Field |
|----------|---------|-----------|-------------|
| DeepSeek | Automatic | 64 | `usage.prompt_cache_hit_tokens` |
| OpenAI | Automatic | 1128 | `usage.prompt_tokens_details.cached_tokens` |
| Claude | Explicit `cache_control` | 1024 | `usage.cache_read_input_tokens` |
| Gemini | Explicit `cachedContent` | 32000 | Response metadata |

## Anti-Patterns

Do NOT:
- Inject `datetime.now()` into the system prompt
- Rebuild the full message list each turn
- Insert retrieved documents in the middle of history
- Append raw tool output with timestamps/UUIDs

## Template

```python
PREFIX = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + DOCS}]
history = []

def chat(user_input):
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="...",
        messages=PREFIX + history,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply
```
