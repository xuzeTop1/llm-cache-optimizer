"""
OpenCode Agent Loop — model-agnostic prefix caching.

OpenCode supports multiple LLM backends (Claude, GPT, DeepSeek, etc.).
The caching strategy depends on the backend:
- Claude backend: use explicit cache_control (see claude_code_example.py)
- OpenAI backend: automatic prefix cache (see codex_example.py)
- DeepSeek backend: automatic prefix cache — just keep prefix stable

This example shows a backend-agnostic pattern that works across all providers.
"""
import os
from openai import OpenAI

# OpenCode typically uses OpenAI-compatible API, even for Claude
# via providers like Anthropic's messages API or OpenRouter
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
)

# ── Compile prefix once at module level ──────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful coding assistant. You can read files, run commands,
and write code. Follow these rules:
1. Use type hints everywhere
2. Prefer composition over inheritance
3. Handle errors explicitly
4. Include docstrings for public functions
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

PREFIX_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

# ── Dynamic history ───────────────────────────────────────────────────────────
dynamic_history: list = []

# ── Summary buffer compaction ────────────────────────────────────────────────
COMPACT_THRESHOLD = 30  # turns before compaction

def compact_history(history: list) -> list:
    """Summarize old turns when the history gets too long."""
    if len(history) < COMPACT_THRESHOLD * 2:
        return history

    old_turns = history[:COMPACT_THRESHOLD]
    recent = history[COMPACT_THRESHOLD:]

    summary_response = client.chat.completions.create(
        model=os.environ.get("MODEL", "gpt-4o"),
        messages=PREFIX_MESSAGES + old_turns + [{
            "role": "user",
            "content": "Summarize the conversation in 3-5 bullet points, "
                       "preserving all key decisions and facts.",
        }],
        max_tokens=200,
    )
    summary = summary_response.choices[0].message.content

    return [{
        "role": "user",
        "content": f"<conversation_summary>\n{summary}\n</conversation_summary>",
    }] + recent


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
        model=os.environ.get("MODEL", "gpt-4o"),
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

    # 5. Log cache metrics (works across OpenAI, DeepSeek, compatible APIs)
    usage = response.usage
    hit = getattr(usage, "prompt_cache_hit_tokens",
                  getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", usage.prompt_tokens - hit)
    total = hit + miss
    rate = hit / total if total else 0
    print(f"[Turn {len(dynamic_history) // 2}] cache={rate:.0%}  hit={hit}  miss={miss}")

    return msg.content or ""


def execute_tool(name: str, args: str) -> str:
    """Stub — replace with real tool execution."""
    import json
    params = json.loads(args)
    return f"Tool {name} executed with {params}"


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(chat("Help me refactor the auth module to use JWT."))
