"""Built-in, dependency-free embedders.

HashEmbedder is a deterministic hashing bag-of-words embedder. It needs no model
download and no service, so retrieval works the instant you ``pip install
attestq`` — great for trying the mechanics, tests, and offline demos.

It is NOT a semantic embedder: it matches on shared tokens, not meaning. For real
assessments use a proper embedder (OpenAIEmbedder, OllamaEmbedder, or any
sentence-transformers model) — they all satisfy the same simple callable contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic hashing bag-of-words embedder.

    Args:
        dim: Output vector dimensionality. Larger reduces hash collisions.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            # Stable across processes (unlike builtin hash()), so results are
            # reproducible and persisted vectors stay valid.
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
