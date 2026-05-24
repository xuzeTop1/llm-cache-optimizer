"""Run a DeepSeek prefix-cache benchmark for naive vs optimized prompts."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_cache_optimizer import CacheAwareClient, CacheMetrics


TURN_COUNT = 30
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "benchmark.csv"
CHART_PATH = OUTPUT_DIR / "benchmark.png"

SYSTEM_PROMPT = """
You are a helpful coding assistant. Use clear Python examples, explain tradeoffs,
and keep stable project context unchanged across turns. The benchmark compares
naive prompt reconstruction with a cache-aware prompt layout.
"""

STATIC_CONTEXT = """
Project: llm-cache-optimizer
Language: Python 3.9+
Build: hatchling
Tests: pytest
Goal: maximize LLM prompt prefix cache hit rates.
"""

QUESTIONS = [
    "Explain prompt prefix caching.",
    "Show a retry pattern.",
    "Design a cache metrics class.",
    "Explain tool schema stability.",
    "Write a memory summarizer.",
]


@dataclass(slots=True)
class TurnResult:
    """Per-turn benchmark result."""

    mode: str
    turn: int
    hit_rate: float
    prompt_tokens: int
    cached_tokens: int
    cost: float


def usage_to_metrics(usage: Any) -> CacheMetrics:
    """Convert a provider usage object into CacheMetrics with DeepSeek pricing."""

    metrics = CacheMetrics.from_provider(MODEL)
    metrics.update_from_usage(usage)
    return metrics


def turn_cost(metrics: CacheMetrics) -> float:
    """Estimate input token cost for one turn."""

    uncached_tokens = max(0, metrics.prompt_tokens - metrics.cached_tokens)
    uncached_cost = uncached_tokens * metrics.input_cost_per_1m / 1_000_000
    cached_cost = metrics.cached_tokens * metrics.cached_input_cost_per_1m / 1_000_000
    return uncached_cost + cached_cost


def run_naive(client: OpenAI) -> list[TurnResult]:
    """Run a naive benchmark that rebuilds system prompt with volatile data."""

    history: list[dict[str, str]] = []
    results: list[TurnResult] = []
    for turn in range(1, TURN_COUNT + 1):
        volatile_system = (
            f"{SYSTEM_PROMPT}\n\n{STATIC_CONTEXT}\n"
            f"Current timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        question = QUESTIONS[(turn - 1) % len(QUESTIONS)]
        history.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": volatile_system}] + history,
        )
        answer = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": answer})

        metrics = usage_to_metrics(response.usage)
        results.append(
            TurnResult(
                mode="naive",
                turn=turn,
                hit_rate=metrics.hit_rate,
                prompt_tokens=metrics.prompt_tokens,
                cached_tokens=metrics.cached_tokens,
                cost=turn_cost(metrics),
            )
        )
    return results


def run_optimized(openai_client: OpenAI) -> list[TurnResult]:
    """Run an optimized benchmark with stable prefix layers."""

    runtime = CacheAwareClient(
        chat_callable=lambda messages, **kwargs: openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **kwargs,
        ),
        metrics=CacheMetrics.from_provider(MODEL),
    )
    runtime.add_core(SYSTEM_PROMPT)
    runtime.add_static_context(STATIC_CONTEXT)

    results: list[TurnResult] = []
    for turn in range(1, TURN_COUNT + 1):
        question = QUESTIONS[(turn - 1) % len(QUESTIONS)]
        response = runtime.chat(question)
        metrics = usage_to_metrics(response.usage)
        results.append(
            TurnResult(
                mode="optimized",
                turn=turn,
                hit_rate=metrics.hit_rate,
                prompt_tokens=metrics.prompt_tokens,
                cached_tokens=metrics.cached_tokens,
                cost=turn_cost(metrics),
            )
        )
    return results


def write_csv(results: list[TurnResult]) -> None:
    """Write per-turn cache metrics to CSV."""

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mode", "turn", "hit_rate", "prompt_tokens", "cached_tokens", "cost"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "mode": result.mode,
                    "turn": result.turn,
                    "hit_rate": f"{result.hit_rate:.6f}",
                    "prompt_tokens": result.prompt_tokens,
                    "cached_tokens": result.cached_tokens,
                    "cost": f"{result.cost:.8f}",
                }
            )


def write_chart(results: list[TurnResult]) -> None:
    """Generate a cache hit-rate chart with matplotlib."""

    import matplotlib.pyplot as plt

    by_mode = {
        "naive": [result for result in results if result.mode == "naive"],
        "optimized": [result for result in results if result.mode == "optimized"],
    }
    plt.figure(figsize=(10, 5))
    for mode, rows in by_mode.items():
        plt.plot(
            [row.turn for row in rows],
            [row.hit_rate * 100 for row in rows],
            marker="o",
            label=mode,
        )
    plt.title("Prompt Cache Hit Rate by Turn")
    plt.xlabel("Turn")
    plt.ylabel("Cache hit rate (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH)


def print_summary(results: list[TurnResult]) -> None:
    """Print average hit rate, total cost, and savings percentage."""

    groups = {
        "naive": [result for result in results if result.mode == "naive"],
        "optimized": [result for result in results if result.mode == "optimized"],
    }
    naive_cost = sum(result.cost for result in groups["naive"])
    optimized_cost = sum(result.cost for result in groups["optimized"])
    savings = 0.0 if naive_cost == 0 else (naive_cost - optimized_cost) / naive_cost

    print("| Metric | Naive | Optimized |")
    print("|---|---:|---:|")
    for mode, rows in groups.items():
        if not rows:
            raise ValueError(f"Missing benchmark rows for {mode}")
    print(
        "| Avg hit rate | "
        f"{_avg_hit_rate(groups['naive']):.1%} | {_avg_hit_rate(groups['optimized']):.1%} |"
    )
    print(f"| Total cost | ${naive_cost:.6f} | ${optimized_cost:.6f} |")
    print(f"| Savings | - | {savings:.1%} |")


def _avg_hit_rate(rows: list[TurnResult]) -> float:
    """Return average hit rate for benchmark rows."""

    return sum(row.hit_rate for row in rows) / len(rows)


def main() -> None:
    """Run the benchmark and write CSV/chart artifacts."""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before running this benchmark.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    results = [*run_naive(client), *run_optimized(client)]
    write_csv(results)
    write_chart(results)
    print_summary(results)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {CHART_PATH}")


if __name__ == "__main__":
    main()
