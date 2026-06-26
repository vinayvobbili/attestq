"""Pluggable interfaces.

attestq is deliberately model-, embedder-, and store-agnostic. You inject the
pieces; the kernel orchestrates them. The two LLM/embedding hooks are plain
callables so wrapping any provider is a one-liner:

    chat = lambda prompt: my_client.complete(prompt)
    embed = lambda texts: my_model.encode(list(texts)).tolist()

VectorStore and Reranker are Protocols so the in-memory default and the optional
Chroma adapter are interchangeable.
"""

from __future__ import annotations

from typing import Callable, List, Protocol, Sequence, runtime_checkable

from .models import Hit

# A chat completion: prompt in, text out. Inject any LLM behind this.
ChatFn = Callable[[str], str]

# An embedder: a batch of texts in, one vector per text out.
EmbedFn = Callable[[Sequence[str]], List[List[float]]]


@runtime_checkable
class VectorStore(Protocol):
    """Stores embedded evidence chunks and answers nearest-neighbour queries.

    Implementations must honour `namespace` so a single store can hold evidence
    for many independent corpora (e.g. one per vendor) without cross-contamination.
    """

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
        namespace: str = "default",
    ) -> None:
        ...

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        namespace: str = "default",
    ) -> List[Hit]:
        ...

    def count(self, namespace: str = "default") -> int:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-orders retrieved hits by relevance to the query.

    A cross-encoder reranker materially improves precision on small corpora,
    where pure vector similarity can drop a single focused document. Optional.
    """

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> List[Hit]:
        ...
