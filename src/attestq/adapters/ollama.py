"""Local Ollama adapters for embeddings and chat.

Run models locally with no API key and no data leaving the host — a common fit
for sensitive evidence (SOC 2 reports, internal policies). Embeddings power
retrieval; the chat adapter is offered for fully-local pipelines.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ._util import require

DEFAULT_HOST = "http://localhost:11434"


class OllamaEmbedder:
    """A ``(texts) -> list[vector]`` embedder backed by a local Ollama server.

    Prefers the batch ``/api/embed`` endpoint and falls back to per-text
    ``/api/embeddings`` for older Ollama versions.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ):
        self._requests = require("requests", "ollama")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        try:
            resp = self._requests.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings:
                return embeddings
        except Exception:
            pass  # fall back to the legacy single-item endpoint
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        resp = self._requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


class OllamaChat:
    """A ``(prompt:str) -> str`` chat callable backed by a local Ollama server."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        system: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 300.0,
    ):
        self._requests = require("requests", "ollama")
        self.model = model
        self.host = host.rstrip("/")
        self.system = system
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.system:
            payload["system"] = self.system
        resp = self._requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
