"""In-memory vector store — the zero-dependency default.

Holds everything in process memory and scores with pure-Python cosine
similarity. Perfect for tests, notebooks, small corpora, and CI. For persistence
or large corpora, install ``attestq[chroma]`` and use ChromaStore instead — it
satisfies the same VectorStore protocol, so swapping is a one-line change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .models import Hit


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0 if either vector is zero-length."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass
class _Record:
    id: str
    text: str
    embedding: List[float]
    metadata: dict


@dataclass
class InMemoryVectorStore:
    """A namespaced, in-memory VectorStore implementation."""

    _records: Dict[str, List[_Record]] = field(default_factory=dict)

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
        namespace: str = "default",
    ) -> None:
        bucket = self._records.setdefault(namespace, [])
        for id_, text, emb, meta in zip(ids, texts, embeddings, metadatas):
            bucket.append(_Record(id=id_, text=text, embedding=list(emb), metadata=dict(meta)))

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        namespace: str = "default",
    ) -> List[Hit]:
        bucket = self._records.get(namespace, [])
        scored = [
            Hit(
                id=r.id,
                text=r.text,
                score=_to_unit(cosine_similarity(embedding, r.embedding)),
                metadata=r.metadata,
            )
            for r in bucket
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def count(self, namespace: str = "default") -> int:
        return len(self._records.get(namespace, []))

    def clear(self, namespace: str | None = None) -> None:
        """Drop a namespace (or everything when namespace is None)."""
        if namespace is None:
            self._records.clear()
        else:
            self._records.pop(namespace, None)


def _to_unit(cos: float) -> float:
    """Clamp cosine to [0, 1] for use as a confidence score.

    Negative similarity means "unrelated", so it floors at 0 rather than mapping
    to the midpoint. This keeps the Engine's ``min_confidence`` gate meaningful:
    a default like 0.45 separates genuinely relevant evidence from noise, the way
    sentence-transformer / Ollama cosine scores naturally distribute.
    """
    return max(0.0, min(1.0, cos))
