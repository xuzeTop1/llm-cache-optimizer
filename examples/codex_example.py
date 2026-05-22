"""
OpenAI Codex CLI Agent Loop — automatic prefix caching.

Codex uses OpenAI's automatic prefix cache. No explicit cache_control needed.
Just keep the prefix stable, and any prefix >=1128 tokens will be cached.

This is the simplest pattern — ideal for getting started.
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Compile prefix once at module level ──────────────────────────────────────
# Must be >=1128 tokens for OpenAI to cache it.
SYSTEM_PROMPT = """
You are a helpful coding assistant built with Codex. Always respond with
clean, production-grade code.

Rules:
1. Use type hints everywhere
2. Prefer composition over inheritance
3. Handle errors explicitly — never use bare except
4. Include docstrings for all public functions
5. Write tests alongside implementation
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to execute"}
                },
                "required": ["cmd"],
            },
        },
    },
]

# PREFIX: static layers concatenated once
PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

# ── Dynamic history ───────────────────────────────────────────────────────────
dynamic_history: list = []

# ── Context compaction (rolling window) ──────────────────────────────────────
MAX_HISTORY_TURNS = 20

def compact_history(history: list) -> list:
    """Keep only the last N turns. Prefix is always preserved."""
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history
    return history[-(MAX_HISTORY_TURNS * 2):]


# ── Agent loop ────────────────────────────────────────────────────────────────
def chat(user_input: str) -> str:
    global dynamic_history

    # 1. Compact if needed (never touches PREFIX)
    dynamic_history = compact_history(dynamic_history)

    # 2. Append user turn
    dynamic_history.append({"role": "user", "content": user_input})

    # 3. PREFIX always first
    messages = PREFIX_MESSAGES + dynamic_history

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    msg = response.choices[0].message

    # 4. Handle tool calls
    if msg.tool_calls:
        dynamic_history.append(msg.model_dump())
        for tc in msg.tool_calls:
            result = execute_tool(tc.function.name, tc.function.arguments)
            dynamic_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        dynamic_history.append({"role": "assistant", "content": msg.content})

    # 5. Log cache metrics
    usage = response.usage
    hit = usage.prompt_tokens_details.cached_tokens or 0
    miss = usage.prompt_tokens - hit
    rate = hit / (hit + miss) if (hit + miss) else 0
    print(f"[Turn {len(dynamic_history)//2}] cache={rate:.0%}  hit={hit}  miss={miss}")

    return msg.content or ""


def execute_tool(name: str, args: str) -> str:
    """Stub — replace with real tool execution."""
    import json
    params = json.loads(args)
    # Real implementation would call actual tools
    return f"Tool {name} executed with {params}"


# ── Cache warm-up for scheduled jobs ─────────────────────────────────────────
def warm_cache():
    """Pre-fire a request to load prefix into cache before real workload."""
    client.chat.completions.create(
        model="gpt-4o",
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warm_cache()
    print(chat("Help me set up a new Next.js project with TypeScript."))
