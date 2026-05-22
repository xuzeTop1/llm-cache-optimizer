"""
Claude Code Agent Loop — with cache_control breakpoints.

This example demonstrates the canonical pattern for Anthropic Claude
with explicit cache_control breakpoints on stable prefix layers.

Key insight: place cache_control on each stable layer boundary,
ensuring each marked block is at least 1024 tokens.
"""
import anthropic

client = anthropic.Anthropic()

# ── Layer 1: Core System (never changes) ─────────────────────────────────────
LAYER_CORE = """
You are a helpful coding assistant. Always respond with clean, production-grade code.
Follow these rules:
1. Use type hints everywhere
2. Prefer composition over inheritance
3. Handle errors explicitly — never use bare except
4. Include docstrings for all public functions
"""

# ── Layer 2: Tool Schemas (changes only on deploy) ───────────────────────────
LAYER_TOOLS = """
Available tools:
- read_file(path: str) -> str: Read a file's contents
- write_file(path: str, content: str) -> bool: Write to a file
- search_code(query: str) -> list[str]: Search the codebase
- run_command(cmd: str) -> str: Execute a shell command
"""

# ── Layer 3: RAG Static (changes per task) ───────────────────────────────────
LAYER_RAG = """
<project_context>
Framework: Next.js 14 + TypeScript
Database: PostgreSQL with Prisma ORM
Testing: Vitest + Playwright
</project_context>
"""

# ── Compile prefix with cache_control at each layer boundary ─────────────────
PREFIX_MESSAGES = [
    {
        "role": "user",
        "content": [
            # Mark each layer as individual cache blocks
            {"type": "text", "text": LAYER_CORE,  "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": LAYER_TOOLS, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": LAYER_RAG,   "cache_control": {"type": "ephemeral"}},
        ],
    },
    # Required: Claude needs an assistant turn after user content
    {"role": "assistant", "content": "Understood. Ready to assist."},
]

# ── Dynamic history (append-only, per turn) ──────────────────────────────────
dynamic_history: list = []

# ── Context compaction ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 30

def compact_history(history: list) -> list:
    """Compact old turns into a summary message, preserving prefix integrity."""
    if len(history) < MAX_HISTORY_TURNS * 2:
        return history

    old_turns = history[:MAX_HISTORY_TURNS]
    recent = history[MAX_HISTORY_TURNS:]

    summary = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system="Summarize the conversation: key decisions, code changes, and open questions.",
        messages=PREFIX_MESSAGES + old_turns,
    )

    return [{"role": "user", "content": f"<summary>\n{summary.content[0].text}\n</summary>"}] + recent


# ── Agent loop ────────────────────────────────────────────────────────────────
def chat(user_input: str) -> str:
    global dynamic_history

    # 1. Compact if needed
    dynamic_history = compact_history(dynamic_history)

    # 2. Append user turn
    dynamic_history.append({"role": "user", "content": user_input})

    # 3. PREFIX always first
    messages = PREFIX_MESSAGES + dynamic_history

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=messages,
    )

    msg = response.content[0].text
    dynamic_history.append({"role": "assistant", "content": msg})

    # 4. Log cache metrics
    usage = response.usage
    hit = usage.cache_read_input_tokens or 0
    miss = usage.input_tokens - hit
    rate = hit / (hit + miss) if (hit + miss) else 0
    print(f"[Turn {len(dynamic_history)//2}] cache={rate:.0%}  hit={hit}  miss={miss}")

    return msg


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(chat("Help me refactor the auth module to use JWT instead of sessions."))
