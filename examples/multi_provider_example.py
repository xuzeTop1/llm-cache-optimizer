"""
Multi-Provider Agent Loop — full 5-layer architecture with PrefixRegistry.

This example demonstrates the complete production pattern:
- 5-layer prompt architecture (core → tools → RAG → session → history)
- PrefixRegistry for multi-worker shared prefixes
- Tool result canonicalization (strip timestamps/UUIDs/volatile fields)
- Summary buffer compaction
- Cache warm-up for scheduled jobs
- Cross-provider cache diagnostics

Works with: DeepSeek, Anthropic Claude, OpenAI, OpenRouter, and any
OpenAI-compatible API.
"""
import os
import json
import re
import hashlib
from typing import Any, Optional
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Provider Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Uncomment your provider:
PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

if PROVIDER == "deepseek":
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
    MODEL = "deepseek-chat"
elif PROVIDER == "openai":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = "gpt-4o"
elif PROVIDER == "openrouter":
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    MODEL = "anthropic/claude-sonnet-4"
else:
    client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ.get("API_BASE_URL"))
    MODEL = os.environ.get("MODEL", "gpt-4o")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: CORE_SYSTEM — Never changes. Cached indefinitely.
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_CORE = """
You are a helpful coding assistant. Follow these principles:

1. Code quality: type hints, docstrings, error handling, tests
2. Architecture: composition over inheritance, single responsibility
3. Communication: be concise, show code over description
4. Safety: never delete files without confirmation, explain risks
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: TOOLS — Changes only on deploy.
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates if not exists)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["cmd"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: RAG_STATIC — Changes per task/session start.
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_RAG = """
<project_context>
Language: Python 3.12+
Framework: FastAPI
Database: PostgreSQL
Testing: pytest + pytest-asyncio
CI/CD: GitHub Actions
</project_context>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Prefix Construction — Layers 1-3 compiled once
# ═══════════════════════════════════════════════════════════════════════════════

PREFIX_MESSAGES = [
    {
        "role": "system",
        "content": "\n\n".join([LAYER_CORE, LAYER_RAG]),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Prefix Hash Registry — Shared across workers
# ═══════════════════════════════════════════════════════════════════════════════

class PrefixRegistry:
    """Central registry for shared prefixes. Workers reference by hash."""

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


PREFIX_KEY = PrefixRegistry.register(PREFIX_MESSAGES)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Result Canonicalization — Strip volatile fields
# ═══════════════════════════════════════════════════════════════════════════════

def canonicalize_tool_result(result: Any) -> str:
    """Normalize tool output to maximize cache stability."""
    if isinstance(result, str):
        # Strip ISO timestamps
        result = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?',
            '<timestamp>',
            result
        )
        return result

    if isinstance(result, dict):
        VOLATILE_KEYS = {
            "timestamp", "ts", "request_id", "trace_id",
            "latency_ms", "created_at", "updated_at", "expires_at", "_debug",
        }
        cleaned = {k: v for k, v in result.items() if k not in VOLATILE_KEYS}

        def _clean(v):
            if isinstance(v, float):
                return round(v, 4)
            if isinstance(v, dict):
                return {k2: _clean(v2) for k2, v2 in v.items()}
            if isinstance(v, list):
                return [_clean(i) for i in v]
            return v

        cleaned = {k: _clean(v) for k, v in cleaned.items()}
        return json.dumps(cleaned, sort_keys=True, ensure_ascii=False)

    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def append_tool_result(history: list, tool_call_id: str, raw_result: Any):
    """Canonicalize before appending to preserve cache continuity."""
    history.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": canonicalize_tool_result(raw_result),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: SESSION_MEMORY — Per-session state
# ═══════════════════════════════════════════════════════════════════════════════

session_memory: dict[str, Any] = {
    "user_name": "Developer",
    "preferred_language": "Python",
    "current_branch": "main",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5: DYNAMIC_HISTORY — Append-only, per turn
# ═══════════════════════════════════════════════════════════════════════════════

dynamic_history: list = []


# ═══════════════════════════════════════════════════════════════════════════════
# Context Compaction — Summary buffer strategy
# ═══════════════════════════════════════════════════════════════════════════════

COMPACT_THRESHOLD = 30

def compact_history(history: list, prefix: list) -> list:
    """Summarize old turns when history exceeds threshold."""
    if len(history) < COMPACT_THRESHOLD * 2:
        return history

    old_turns = history[:COMPACT_THRESHOLD]
    recent = history[COMPACT_THRESHOLD:]

    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=prefix + old_turns + [{
            "role": "user",
            "content": "Summarize the conversation so far in 3-5 bullet points, "
                       "preserving all key decisions and facts.",
        }],
        max_tokens=200,
    )
    summary_text = summary_response.choices[0].message.content

    return [{
        "role": "user",
        "content": f"<conversation_summary>\n{summary_text}\n</conversation_summary>",
    }] + recent


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def log_cache(usage, turn: int):
    # DeepSeek / OpenAI-compatible
    hit = getattr(usage, "prompt_cache_hit_tokens", 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", 0)

    # OpenAI nested format
    if hasattr(usage, "prompt_tokens_details"):
        hit = usage.prompt_tokens_details.cached_tokens or 0
        miss = usage.prompt_tokens - hit

    total = hit + miss
    rate = hit / total if total else 0
    print(f"[Turn {turn}] cache={rate:.0%}  hit={hit}  miss={miss}  total={total}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Warm-up
# ═══════════════════════════════════════════════════════════════════════════════

def warm_cache():
    """Pre-fire cache before real workload starts."""
    client.chat.completions.create(
        model=MODEL,
        messages=PREFIX_MESSAGES + [{"role": "user", "content": "ping"}],
        max_tokens=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Worker
# ═══════════════════════════════════════════════════════════════════════════════

class AgentWorker:
    """Worker that uses shared prefix via PrefixRegistry."""

    def __init__(self, prefix_key: str):
        self.prefix = PrefixRegistry.get(prefix_key)  # shared reference
        self.history: list = []                         # worker-local

    def run(self, user_input: str) -> str:
        # 1. Compact if needed (never touches prefix)
        self.history = compact_history(self.history, self.prefix)

        # 2. Append user turn
        self.history.append({"role": "user", "content": user_input})

        # 3. PREFIX always first
        messages = self.prefix + self.history

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        msg = response.choices[0].message

        # 4. Handle tool calls with canonicalization
        if msg.tool_calls:
            self.history.append(msg.model_dump())
            for tc in msg.tool_calls:
                raw = execute_tool(tc.function.name, tc.function.arguments)
                append_tool_result(self.history, tc.id, raw)
        else:
            self.history.append({"role": "assistant", "content": msg.content})

        # 5. Log cache metrics
        log_cache(response.usage, len(self.history) // 2)

        return msg.content or ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Execution Stub
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, args: str) -> str:
    """Replace with real tool implementations."""
    params = json.loads(args)
    return json.dumps({"status": "ok", "tool": name, "params": params})


# ═══════════════════════════════════════════════════════════════════════════════
# Usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Warm the cache
    warm_cache()

    # Create a worker using the shared prefix
    worker = AgentWorker(prefix_key=PREFIX_KEY)

    # Simulate conversation
    print(worker.run("What does llm-cache-optimizer do?"))
    print(worker.run("Show me how to add cache_control breakpoints."))
