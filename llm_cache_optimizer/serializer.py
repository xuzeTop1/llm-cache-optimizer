"""Canonical serialization for cache-stable prompts and tool results."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "ts",
        "time",
        "now",
        "date",
        "created_at",
        "updated_at",
        "expires_at",
        "request_id",
        "trace_id",
        "span_id",
        "run_id",
        "uuid",
        "idempotency_key",
        "nonce",
        "_debug",
    }
)

ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
REQUEST_ID_RE = re.compile(r"\b(?:req|request|trace|span|run)_[A-Za-z0-9_-]{8,}\b")


@dataclass(slots=True)
class CanonicalSerializer:
    """Normalize prompt inputs so equivalent content produces stable strings."""

    volatile_keys: set[str] = field(default_factory=lambda: set(DEFAULT_VOLATILE_KEYS))
    timestamp_placeholder: str = "<timestamp>"
    uuid_placeholder: str = "<uuid>"
    request_id_placeholder: str = "<request_id>"
    sort_keys: bool = True
    ensure_ascii: bool = False

    def normalize(self, prompt: Any) -> str:
        """Return a stable string representation for strings, mappings, and lists."""

        cleaned = self.clean(prompt)
        if isinstance(cleaned, str):
            return self._normalize_text(cleaned)
        return json.dumps(
            cleaned,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
            separators=(",", ":"),
        )

    def clean(self, value: Any) -> Any:
        """Remove volatile fields and normalize nested values."""

        if isinstance(value, str):
            return self._normalize_text(value)

        if isinstance(value, Mapping):
            normalized_items = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text.lower() in self.volatile_keys:
                    continue
                normalized_items[key_text] = self.clean(item)
            return dict(sorted(normalized_items.items()))

        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [self.clean(item) for item in value]

        if isinstance(value, float):
            return round(value, 6)

        return value

    def _normalize_text(self, text: str) -> str:
        text = ISO_TIMESTAMP_RE.sub(self.timestamp_placeholder, text)
        text = UUID_RE.sub(self.uuid_placeholder, text)
        text = REQUEST_ID_RE.sub(self.request_id_placeholder, text)
        return " ".join(text.split())
