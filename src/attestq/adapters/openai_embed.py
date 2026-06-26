"""OpenAI-compatible embeddings adapter.

Pairs with OpenAIChat to make OpenAI (or any OpenAI-compatible endpoint) a
complete provider for attestq. Like the chat adapter, ``base_url`` lets you point
it at Azure, a local server, or a corporate gateway.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ._util import require


class OpenAIEmbedder:
    """A ``(texts) -> list[vector]`` embedder backed by an OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        batch_size: int = 256,
    ):
        openai = require("openai", "openai")
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = openai.OpenAI(**kwargs)
        self.model = model
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # The API preserves input order; sort defensively just in case.
            data = sorted(resp.data, key=lambda d: d.index)
            out.extend(d.embedding for d in data)
        return out
