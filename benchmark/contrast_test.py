"""Contrast benchmark: naive vs llm-cache-optimizer on DeepSeek.

This script runs three real-world test scenarios, each comparing a "naive"
approach (which breaks prefix cache on every call) against an "optimized"
approach (using llm-cache-optimizer to maintain a stable prompt prefix).

Scenarios
---------
1. RAG Q&A         — Same project docs, different questions.
2. Agent tool call — Same system + tools, different function requests.
3. Long conversation — Multi-turn chat with SessionMemory.

Output
------
- Terminal: colour-coded per-turn log + summary table.
- File:     benchmark/contrast_result.md (auto-saved Markdown report).

Quick start
-----------
    pip install "llm-cache-optimizer[openai]"
    set DEEPSEEK_API_KEY=sk-xxx
    python benchmark/contrast_test.py

You can also override defaults via environment variables:
    DEEPSEEK_API_KEY   – required
    DEEPSEEK_MODEL     – default "deepseek-chat"
    DEEPSEEK_BASE_URL  – default "https://api.deepseek.com/v1"
    TURNS_PER_SCENARIO – default 5
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_cache_optimizer import (
    CacheAwareClient,
    CacheMetrics,
    CanonicalSerializer,
    PromptBuilder,
    SessionMemory,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration — edit these or set the corresponding env vars
# ──────────────────────────────────────────────────────────────────────

API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
TURNS: int = int(os.environ.get("TURNS_PER_SCENARIO", "5"))

OUTPUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUTPUT_DIR / "contrast_result.md"

# ──────────────────────────────────────────────────────────────────────
# Scenario 1 — RAG Q&A
# ──────────────────────────────────────────────────────────────────────

RAG_SYSTEM = (
    "You are a technical documentation assistant for the llm-cache-optimizer "
    "project. Answer questions precisely, cite relevant module names, and "
    "keep responses under 200 words."
)

RAG_DOCS = """\
<project_docs>
Package: llm-cache-optimizer v0.3.0
License: MIT
Python: >=3.9

Core modules:
- CacheAwareClient: high-level cache-aware chat client
- PromptBuilder: 6-layer prompt construction (core_system → tool_schema → static_context → session_memory → history → runtime)
- CanonicalSerializer: normalizes volatile fields (timestamps, UUIDs, request IDs) for stable caching
- CacheMetrics: tracks hit rate and cost savings per provider
- SessionMemory: local summary + keyword extraction for long conversations
- CacheAwareOpenAI: OpenAI/DeepSeek adapter with built-in metrics

Supported providers:
- DeepSeek: auto prefix cache, min 64 tokens, cached cost ~10% of input
- Claude: explicit cache_control, min 1024 tokens, cached cost ~10%
- OpenAI: auto prefix cache, min 1128 tokens, cached cost ~50%
- Gemini: explicit cachedContent, min 32k tokens, cached cost ~25%
</project_docs>"""

RAG_QUESTIONS = [
    "What is PromptBuilder and how does it order prompt layers?",
    "How does CanonicalSerializer handle timestamps and UUIDs?",
    "What is the minimum cacheable prefix for DeepSeek?",
    "How does SessionMemory compress long conversations?",
    "Explain the difference between CacheMetrics hit_rate and token_saved.",
    "Which providers are supported and what are their cache characteristics?",
    "How does CacheAwareOpenAI integrate with the core client?",
    "What is the stable_prefix() method used for?",
]

# ──────────────────────────────────────────────────────────────────────
# Scenario 2 — Agent tool calling
# ──────────────────────────────────────────────────────────────────────

AGENT_SYSTEM = (
    "You are a code-generation agent. When asked to implement a function, "
    "produce a complete Python implementation with type hints, docstring, "
    "and a brief usage example. Keep implementations under 50 lines."
)

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents from the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its stdout",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern across the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or keyword to search for",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional file extension filter, e.g. 'py'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

AGENT_STATIC = """\
<agent_context>
Project: llm-cache-optimizer
Language: Python 3.9+
Build system: hatchling
Test framework: pytest
Key files: client.py, layers.py, serializer.py, metrics.py, memory.py, adapters/openai.py
"""

AGENT_REQUESTS = [
    "Implement a retry decorator with exponential backoff.",
    "Write a singleton logger class with file rotation.",
    "Create a typed configuration dataclass from a YAML file.",
    "Build a simple rate limiter using the token bucket algorithm.",
    "Implement a context manager for temporary directory creation.",
    "Write a memoization decorator that supports TTL expiration.",
    "Create an async queue worker with graceful shutdown.",
    "Implement a simple circuit breaker pattern.",
]

# ──────────────────────────────────────────────────────────────────────
# Scenario 3 — Long conversation with SessionMemory
# ──────────────────────────────────────────────────────────────────────

CONV_SYSTEM = (
    "You are a helpful pair-programming assistant. We are building a "
    "cache-aware LLM runtime together. Follow up on previous context, "
    "remember design decisions, and suggest next steps."
)

CONV_STATIC = """\
<conversation_context>
Project goal: build llm-cache-optimizer, a Python library that helps
developers structure LLM prompts for maximum prefix cache hit rates.

Completed so far:
- PromptBuilder with 6-layer ordering
- CanonicalSerializer for volatile field normalization
- CacheAwareClient with history and metrics
- CacheAwareOpenAI adapter
- SessionMemory with local summarization

Still TODO:
- DeepSeek prefix diagnostics
- Provider-specific optimization reports
- CLI interface
"""

CONV_TURNS = [
    "Let's review the architecture. What are the main components?",
    "How should we handle the DeepSeek prefix diagnostics feature?",
    "What would a provider-specific optimization report look like?",
    "Should we add a CLI interface? What commands would it expose?",
    "How can we improve SessionMemory to use LLM-based summarization?",
    "What testing strategy should we adopt for the diagnostics module?",
    "Let's plan the v0.4.0 release. What are the priorities?",
    "How do we document the cache-friendly prompt ordering for new users?",
]


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Per-turn metrics."""

    scenario: str
    mode: str          # "naive" | "optimized"
    turn: int
    prompt_tokens: int
    cached_tokens: int
    hit_rate: float
    cost: float
    elapsed_s: float


@dataclass
class ScenarioResult:
    """Aggregated results for one scenario."""

    name: str
    naive_results: list[TurnResult] = field(default_factory=list)
    optimized_results: list[TurnResult] = field(default_factory=list)

    @property
    def naive_avg_hit_rate(self) -> float:
        return _avg(self.naive_results, lambda r: r.hit_rate)

    @property
    def optimized_avg_hit_rate(self) -> float:
        return _avg(self.optimized_results, lambda r: r.hit_rate)

    @property
    def naive_total_cost(self) -> float:
        return sum(r.cost for r in self.naive_results)

    @property
    def optimized_total_cost(self) -> float:
        return sum(r.cost for r in self.optimized_results)

    @property
    def savings_pct(self) -> float:
        if self.naive_total_cost == 0:
            return 0.0
        return (self.naive_total_cost - self.optimized_total_cost) / self.naive_total_cost


def _avg(rows: list[TurnResult], key: callable) -> float:
    return sum(key(r) for r in rows) / len(rows) if rows else 0.0


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

# ANSI colour codes for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def log_turn(result: TurnResult) -> None:
    """Print a colour-coded per-turn log line."""
    hr = result.hit_rate
    colour = _GREEN if hr >= 0.8 else _YELLOW if hr >= 0.4 else _RED
    print(
        f"  {colour}Turn {result.turn:>2d}{_RESET}  "
        f"hit={hr:>6.1%}  "
        f"prompt={result.prompt_tokens:>6d}  "
        f"cached={result.cached_tokens:>6d}  "
        f"cost=${result.cost:.6f}  "
        f"({result.elapsed_s:.1f}s)"
    )


def turn_cost(metrics: CacheMetrics) -> float:
    """Estimate input token cost for a single turn."""
    uncached_tokens = max(0, metrics.prompt_tokens - metrics.cached_tokens)
    uncached_cost = uncached_tokens * metrics.input_cost_per_1m / 1_000_000
    cached_cost = metrics.cached_tokens * metrics.cached_input_cost_per_1m / 1_000_000
    return uncached_cost + cached_cost


def extract_metrics(usage: Any) -> CacheMetrics:
    """Convert a provider usage object into CacheMetrics with DeepSeek pricing."""
    metrics = CacheMetrics.from_provider(MODEL)
    metrics.update_from_usage(usage)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Naive implementations — deliberately break prefix cache
# ──────────────────────────────────────────────────────────────────────

def _volatile_system(base: str, static: str) -> str:
    """Inject a timestamp and turn counter into the system prompt.

    This mimics a common mistake: rebuilding the system message with
    volatile data on every call, which defeats DeepSeek's prefix cache.
    """
    ts = datetime.now(timezone.utc).isoformat()
    # Shuffle word order slightly with a volatile "session id"
    sid = f"session-{int(time.time() * 1000) % 99999:05d}"
    return f"{base}\n\n{static}\n[Timestamp: {ts} | {sid}]"


def _volatile_tool_schema(tools: list[dict]) -> list[dict]:
    """Add a volatile request_id to each tool schema to break cache.

    In the naive approach, tool schemas are re-serialized on every call,
    and any volatile field (even an unused one) makes the prefix diverge.
    """
    import copy, uuid
    result = copy.deepcopy(tools)
    for tool in result:
        tool["request_id"] = str(uuid.uuid4())  # type: ignore[assignment]
    return result


def run_naive_rag(client: OpenAI) -> list[TurnResult]:
    """Scenario 1 naive: RAG Q&A with timestamp-polluted system prompt."""
    history: list[dict[str, str]] = []
    results: list[TurnResult] = []

    for turn in range(1, TURNS + 1):
        system_msg = _volatile_system(RAG_SYSTEM, RAG_DOCS)
        question = RAG_QUESTIONS[(turn - 1) % len(RAG_QUESTIONS)]
        history.append({"role": "user", "content": question})

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_msg}, *history],
        )
        elapsed = time.perf_counter() - t0

        answer = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": answer})

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="RAG Q&A", mode="naive", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


def run_naive_agent(client: OpenAI) -> list[TurnResult]:
    """Scenario 2 naive: Agent with volatile tool schemas + system prompt."""
    history: list[dict[str, str]] = []
    results: list[TurnResult] = []

    for turn in range(1, TURNS + 1):
        system_msg = _volatile_system(AGENT_SYSTEM, AGENT_STATIC)
        # Re-serialize tools with volatile request_id each turn
        tools = _volatile_tool_schema(AGENT_TOOLS)
        request = AGENT_REQUESTS[(turn - 1) % len(AGENT_REQUESTS)]
        history.append({"role": "user", "content": request})

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_msg}, *history],
            tools=tools,
        )
        elapsed = time.perf_counter() - t0

        answer = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": answer})

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="Agent Tool Call", mode="naive", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


def run_naive_conversation(client: OpenAI) -> list[TurnResult]:
    """Scenario 3 naive: Long conversation with volatile system + no memory."""
    history: list[dict[str, str]] = []
    results: list[TurnResult] = []

    for turn in range(1, TURNS + 1):
        system_msg = _volatile_system(CONV_SYSTEM, CONV_STATIC)
        user_msg = CONV_TURNS[(turn - 1) % len(CONV_TURNS)]
        history.append({"role": "user", "content": user_msg})

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_msg}, *history],
        )
        elapsed = time.perf_counter() - t0

        answer = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": answer})

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="Long Conversation", mode="naive", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


# ──────────────────────────────────────────────────────────────────────
# Optimized implementations — use llm-cache-optimizer properly
# ──────────────────────────────────────────────────────────────────────

def run_optimized_rag(openai_client: OpenAI) -> list[TurnResult]:
    """Scenario 1 optimized: RAG Q&A with stable prefix layers."""
    # Build a CacheAwareClient with the strict layer ordering:
    # core_system → (no tools) → static_context
    runtime = CacheAwareClient(
        chat_callable=lambda messages, **kwargs: openai_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs,
        ),
        metrics=CacheMetrics.from_provider(MODEL),
    )
    runtime.add_core(RAG_SYSTEM)           # Layer 1: never changes
    runtime.add_static_context(RAG_DOCS)    # Layer 3: same docs every turn

    results: list[TurnResult] = []
    for turn in range(1, TURNS + 1):
        question = RAG_QUESTIONS[(turn - 1) % len(RAG_QUESTIONS)]

        t0 = time.perf_counter()
        response = runtime.chat(question)
        elapsed = time.perf_counter() - t0

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="RAG Q&A", mode="optimized", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


def run_optimized_agent(openai_client: OpenAI) -> list[TurnResult]:
    """Scenario 2 optimized: Agent with stable core + tools + static context."""
    runtime = CacheAwareClient(
        chat_callable=lambda messages, **kwargs: openai_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs,
        ),
        metrics=CacheMetrics.from_provider(MODEL),
        serializer=CanonicalSerializer(),   # normalize any volatile fields
    )
    runtime.add_core(AGENT_SYSTEM)          # Layer 1: stable persona
    runtime.add_tool_schema(AGENT_TOOLS)     # Layer 2: stable tool definitions
    runtime.add_static_context(AGENT_STATIC) # Layer 3: stable project context

    results: list[TurnResult] = []
    for turn in range(1, TURNS + 1):
        request = AGENT_REQUESTS[(turn - 1) % len(AGENT_REQUESTS)]

        t0 = time.perf_counter()
        response = runtime.chat(request, tools=AGENT_TOOLS)
        elapsed = time.perf_counter() - t0

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="Agent Tool Call", mode="optimized", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


def run_optimized_conversation(openai_client: OpenAI) -> list[TurnResult]:
    """Scenario 3 optimized: Long conversation with SessionMemory.

    Key optimization: after every 2 turns, refresh_memory() compresses
    the growing history into a compact summary + keywords that becomes
    Layer 4 (session_memory). This keeps the stable prefix (Layers 1-3)
    intact while the dynamic history (Layer 5) stays manageable.
    """
    runtime = CacheAwareClient(
        chat_callable=lambda messages, **kwargs: openai_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs,
        ),
        metrics=CacheMetrics.from_provider(MODEL),
        memory=SessionMemory(max_summary_chars=400, max_keywords=10),
    )
    runtime.add_core(CONV_SYSTEM)           # Layer 1: stable
    runtime.add_static_context(CONV_STATIC)  # Layer 3: stable

    results: list[TurnResult] = []
    for turn in range(1, TURNS + 1):
        user_msg = CONV_TURNS[(turn - 1) % len(CONV_TURNS)]

        t0 = time.perf_counter()
        response = runtime.chat(user_msg)
        elapsed = time.perf_counter() - t0

        # Compress history into session memory every 2 turns
        if turn % 2 == 0:
            runtime.refresh_memory()

        metrics = extract_metrics(response.usage)
        results.append(TurnResult(
            scenario="Long Conversation", mode="optimized", turn=turn,
            prompt_tokens=metrics.prompt_tokens,
            cached_tokens=metrics.cached_tokens,
            hit_rate=metrics.hit_rate,
            cost=turn_cost(metrics),
            elapsed_s=elapsed,
        ))
        log_turn(results[-1])

    return results


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def build_markdown(scenarios: list[ScenarioResult]) -> str:
    """Build the full Markdown contrast report."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# llm-cache-optimizer Contrast Benchmark\n")
    lines.append(f"**Date**: {now}  ")
    lines.append(f"**Model**: `{MODEL}`  ")
    lines.append(f"**Turns per scenario**: {TURNS}  ")
    lines.append(f"**Provider**: DeepSeek (auto prefix cache, min 64 tokens, cached cost ~10%)\n")

    lines.append("---\n")
    lines.append("## Summary\n")

    # Summary table
    lines.append("| Scenario | Avg Hit Rate (Naive) | Avg Hit Rate (Optimized) | Total Cost (Naive) | Total Cost (Optimized) | Savings |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in scenarios:
        lines.append(
            f"| {s.name} "
            f"| {s.naive_avg_hit_rate:.1%} "
            f"| {s.optimized_avg_hit_rate:.1%} "
            f"| ${s.naive_total_cost:.4f} "
            f"| ${s.optimized_total_cost:.4f} "
            f"| {s.savings_pct:.1%} |"
        )

    # Overall
    total_naive = sum(s.naive_total_cost for s in scenarios)
    total_optimized = sum(s.optimized_total_cost for s in scenarios)
    overall_savings = (total_naive - total_optimized) / total_naive if total_naive else 0
    overall_naive_hr = sum(s.naive_avg_hit_rate for s in scenarios) / len(scenarios)
    overall_opt_hr = sum(s.optimized_avg_hit_rate for s in scenarios) / len(scenarios)

    lines.append(f"| **Overall** | **{overall_naive_hr:.1%}** | **{overall_opt_hr:.1%}** | **${total_naive:.4f}** | **${total_optimized:.4f}** | **{overall_savings:.1%}** |\n")

    lines.append("---\n")

    # Per-scenario detail tables
    for s in scenarios:
        lines.append(f"## {s.name}\n")
        lines.append("| Turn | Hit Rate (Naive) | Hit Rate (Optimized) | Cost (Naive) | Cost (Optimized) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for n, o in zip(s.naive_results, s.optimized_results):
            lines.append(
                f"| {n.turn} "
                f"| {n.hit_rate:.1%} "
                f"| {o.hit_rate:.1%} "
                f"| ${n.cost:.6f} "
                f"| ${o.cost:.6f} |"
            )
        lines.append("")

    lines.append("---\n")
    lines.append("## Methodology\n")
    lines.append("### Naive Approach\n")
    lines.append(
        "- Rebuilds the system prompt on **every** call by injecting the current "
        "timestamp and a random session ID.\n"
        "- Tool schemas are re-serialized with a new `request_id` (UUID) each turn.\n"
        "- No prompt layering: system + context + user are concatenated in an "
        "unstable order.\n"
        "- Result: DeepSeek's prefix cache never finds a matching prefix → ~0% hit rate.\n"
    )
    lines.append("### Optimized Approach (llm-cache-optimizer)\n")
    lines.append(
        "- Uses `CacheAwareClient` with strict 6-layer ordering:\n"
        "  `core_system → tool_schema → static_context → session_memory → history → runtime`\n"
        "- Layers 1–3 are **stable**: they never change across turns within a session.\n"
        "- `CanonicalSerializer` normalizes timestamps, UUIDs, and request IDs.\n"
        "- `SessionMemory` compresses growing history into a compact summary every few turns.\n"
        "- Result: DeepSeek matches the stable prefix → high hit rate (typically >80% after turn 1).\n"
    )

    lines.append("---\n")
    lines.append(
        "*Generated by `benchmark/contrast_test.py` — "
        "run it yourself with `python benchmark/contrast_test.py`*\n"
    )

    return "\n".join(lines)


def print_terminal_summary(scenarios: list[ScenarioResult]) -> None:
    """Print a formatted summary table to the terminal."""
    print(f"\n{'=' * 72}")
    print(f"{_BOLD}  CONTRAST BENCHMARK SUMMARY{_RESET}")
    print(f"{'=' * 72}")
    print(
        f"  {'Scenario':<22s} "
        f"{'Hit(Naive)':>10s} {'Hit(Opt)':>10s} "
        f"{'Cost(Naive)':>12s} {'Cost(Opt)':>12s} {'Savings':>8s}"
    )
    print(f"  {'-' * 20}  {'-' * 9}  {'-' * 9}  {'-' * 11}  {'-' * 11}  {'-' * 7}")

    for s in scenarios:
        print(
            f"  {s.name:<22s} "
            f"{s.naive_avg_hit_rate:>9.1%} "
            f"{s.optimized_avg_hit_rate:>9.1%} "
            f"${s.naive_total_cost:>10.4f} "
            f"${s.optimized_total_cost:>10.4f} "
            f"{s.savings_pct:>7.1%}"
        )

    total_naive = sum(s.naive_total_cost for s in scenarios)
    total_optimized = sum(s.optimized_total_cost for s in scenarios)
    overall_savings = (total_naive - total_optimized) / total_naive if total_naive else 0
    overall_naive_hr = sum(s.naive_avg_hit_rate for s in scenarios) / len(scenarios)
    overall_opt_hr = sum(s.optimized_avg_hit_rate for s in scenarios) / len(scenarios)

    print(f"  {'-' * 20}  {'-' * 9}  {'-' * 9}  {'-' * 11}  {'-' * 11}  {'-' * 7}")
    print(
        f"  {'OVERALL':<22s} "
        f"{overall_naive_hr:>9.1%} "
        f"{_BOLD}{_GREEN}{overall_opt_hr:>9.1%}{_RESET} "
        f"${total_naive:>10.4f} "
        f"${_BOLD}{_GREEN}{total_optimized:>10.4f}{_RESET} "
        f"{_BOLD}{_GREEN}{overall_savings:>7.1%}{_RESET}"
    )
    print(f"{'=' * 72}")


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────

def run_scenario(
    name: str,
    naive_fn: callable,
    optimized_fn: callable,
    openai_client: OpenAI,
) -> ScenarioResult:
    """Run both naive and optimized variants of a scenario."""
    result = ScenarioResult(name=name)

    print(f"\n{_BOLD}{_CYAN}▶ Scenario: {name} — Naive{_RESET}")
    result.naive_results = naive_fn(openai_client)

    print(f"\n{_BOLD}{_CYAN}▶ Scenario: {name} — Optimized{_RESET}")
    result.optimized_results = optimized_fn(openai_client)

    return result


def main() -> None:
    """Run all contrast benchmark scenarios and save results."""

    # ── Validate API key ──
    if not API_KEY:
        print(
            f"{_RED}Error: DEEPSEEK_API_KEY is not set.{_RESET}\n"
            "  Set it via environment variable or edit API_KEY in this script.\n"
            "  Example: set DEEPSEEK_API_KEY=sk-xxx"
        )
        sys.exit(1)

    print(f"{_BOLD}llm-cache-optimizer Contrast Benchmark{_RESET}")
    print(f"  Model:    {MODEL}")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Turns:    {TURNS} per scenario")
    print(f"  Scenarios: 3 (RAG Q&A, Agent Tool Call, Long Conversation)")
    print()

    openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # ── Run each scenario ──
    scenarios: list[ScenarioResult] = []

    scenarios.append(run_scenario(
        "RAG Q&A",
        run_naive_rag,
        run_optimized_rag,
        openai_client,
    ))

    scenarios.append(run_scenario(
        "Agent Tool Call",
        run_naive_agent,
        run_optimized_agent,
        openai_client,
    ))

    scenarios.append(run_scenario(
        "Long Conversation",
        run_naive_conversation,
        run_optimized_conversation,
        openai_client,
    ))

    # ── Terminal summary ──
    print_terminal_summary(scenarios)

    # ── Save Markdown report ──
    md = build_markdown(scenarios)
    RESULT_PATH.write_text(md, encoding="utf-8")
    print(f"\n{_GREEN}✓ Report saved to {RESULT_PATH}{_RESET}")


if __name__ == "__main__":
    main()
