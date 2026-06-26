"""OpenAI-compatible chat adapter.

Works with any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, vLLM, LM Studio,
Ollama's OpenAI-compatible server, or a corporate gateway — just point `base_url`
at it. The instance is callable, so it drops straight into ``Engine(chat=...)``.
"""

from __future__ import annotations

from typing import Optional

from ._util import require


class OpenAIChat:
    """A ``(prompt:str) -> str`` chat callable backed by an OpenAI-compatible API.

    Args:
        model: Model name/deployment to call.
        api_key: API key (falls back to the client's own env handling if None).
        base_url: Override for non-OpenAI compatible endpoints.
        system: Optional system message prepended to every call. attestq's default
            prompt already embeds its instruction, so this is usually left None.
        temperature: Sampling temperature; 0 for deterministic extraction.
        timeout: Per-request timeout in seconds.
        max_tokens: Optional cap on the completion length.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_tokens: Optional[int] = None,
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
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
