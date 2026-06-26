"""Persistent ChromaDB vector store adapter.

Satisfies the same VectorStore protocol as InMemoryVectorStore, so swapping in
persistence is a one-line change. Namespaces map to a metadata filter on a single
collection, keeping each corpus (e.g. each vendor) isolated.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..models import Hit
from ._util import require


class ChromaStore:
    """A persistent VectorStore backed by ChromaDB.

    Args:
        path: On-disk directory for a PersistentClient. If None, an ephemeral
            in-process client is used.
        collection: Collection name (one collection holds all namespaces).
        client: Pass a pre-built chromadb client to reuse a connection.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        collection: str = "attestq_evidence",
        client=None,
    ):
        chromadb = require("chromadb", "chroma")
        if client is not None:
            self._client = client
        elif path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()
        # Cosine space so distances convert cleanly to a [0,1] similarity score,
        # keeping the Engine confidence gate meaningful.
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
        namespace: str = "default",
    ) -> None:
        metas = [{**dict(m), "_namespace": namespace} for m in metadatas]
        self._collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=metas,
        )

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        namespace: str = "default",
    ) -> List[Hit]:
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            where={"_namespace": namespace},
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: List[Hit] = []
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = dict(meta or {})
            meta.pop("_namespace", None)
            hits.append(
                Hit(id=id_, text=doc, score=_cosine_distance_to_score(dist), metadata=meta)
            )
        return hits

    def count(self, namespace: str = "default") -> int:
        res = self._collection.get(where={"_namespace": namespace}, include=[])
        return len(res.get("ids") or [])

    def clear(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self._collection.delete(where={})
        else:
            self._collection.delete(where={"_namespace": namespace})


def _cosine_distance_to_score(distance) -> float:
    """Chroma cosine distance (1 - cosine_sim) -> [0,1] similarity score."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))
