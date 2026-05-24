"""OpenAI SDK adapter for cache-aware chat completions."""

from __future__ import annotations

from typing import Any

from ..client import CacheAwareClient
from ..metrics import CacheMetrics
from ..serializer import CanonicalSerializer


class CacheAwareOpenAI(CacheAwareClient):
    """Cache-aware wrapper around ``openai.OpenAI().chat.completions.create``."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        client: Any | None = None,
        serializer: CanonicalSerializer | None = None,
        metrics: CacheMetrics | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.openai_client = client or self._build_client(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )
        super().__init__(
            chat_callable=self._chat_completion,
            serializer=serializer,
            metrics=metrics,
        )

    def _build_client(
        self,
        api_key: str | None,
        base_url: str | None,
        **client_kwargs: Any,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "CacheAwareOpenAI requires the OpenAI SDK. "
                "Install it with `pip install llm-cache-optimizer[openai]`."
            ) from exc

        kwargs = dict(client_kwargs)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        request = dict(kwargs)
        request.setdefault("model", self.model)
        request["messages"] = messages
        return self.openai_client.chat.completions.create(**request)

    def chat(self, user_input: str, **kwargs: Any) -> Any:
        """Send a cache-aware chat completion request through the OpenAI SDK."""

        return super().chat(user_input, **kwargs)
