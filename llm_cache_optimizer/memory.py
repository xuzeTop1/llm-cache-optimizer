"""Local session memory summarization and keyword extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "best",
        "but",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "into",
        "is",
        "it",
        "need",
        "next",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "we",
        "with",
        "you",
        "your",
        "assistant",
        "system",
        "user",
        "added",
        "summarizes",
        "一个",
        "一下",
        "可以",
        "我们",
        "这个",
        "还是",
        "就是",
        "需要",
        "继续",
    }
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(slots=True)
class SessionMemory:
    """Summarize chat history into a compact cache-friendly memory block."""

    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    max_summary_chars: int = 360
    max_keywords: int = 8
    stopwords: set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))

    def update(self, history: Iterable[dict[str, Any]]) -> "SessionMemory":
        """Refresh summary and keywords from role/content messages."""

        text = _history_text(history)
        self.summary = self.summarize(text)
        self.keywords = self.extract_keywords(text)
        return self

    def summarize(self, text: str) -> str:
        """Create a deterministic local summary without calling an LLM."""

        normalized = " ".join(text.split())
        if len(normalized) <= self.max_summary_chars:
            return normalized

        sentences = [item.strip() for item in SENTENCE_RE.split(normalized) if item.strip()]
        if not sentences:
            return normalized[: self.max_summary_chars].rstrip() + "..."

        selected: list[str] = []
        total = 0
        for sentence in sentences:
            next_total = total + len(sentence) + (1 if selected else 0)
            if next_total > self.max_summary_chars:
                break
            selected.append(sentence)
            total = next_total

        if not selected:
            return sentences[0][: self.max_summary_chars].rstrip() + "..."

        summary = " ".join(selected)
        if len(summary) < len(normalized):
            summary = summary.rstrip(".。") + "..."
        return summary

    def extract_keywords(self, text: str) -> list[str]:
        """Extract high-signal words and short phrases with simple frequency scoring."""

        tokens = []
        for token in TOKEN_RE.findall(text):
            normalized = token.lower()
            if normalized in self.stopwords:
                continue
            tokens.append(normalized)

        counts = Counter(tokens)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _count in ranked[: self.max_keywords]]

    def to_layer(self) -> dict[str, Any]:
        """Return a structured layer suitable for ``add_session_memory``."""

        return {
            "summary": self.summary,
            "keywords": self.keywords,
        }

    def to_text(self) -> str:
        """Return a compact human-readable memory block."""

        keywords = ", ".join(self.keywords)
        return f"Summary: {self.summary}\nKeywords: {keywords}".strip()


def _history_text(history: Iterable[dict[str, Any]]) -> str:
    chunks = []
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if content:
            chunks.append(f"{role}: {content}")
    return "\n".join(chunks)
