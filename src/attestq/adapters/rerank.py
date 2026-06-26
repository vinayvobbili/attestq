"""Cross-encoder reranker adapter.

A cross-encoder scores (query, chunk) pairs jointly and is far more precise than
the bi-encoder similarity used for first-pass retrieval. Using it materially
improves which evidence reaches the LLM — especially on small corpora where one
focused document is easy to miss.

Raw cross-encoder outputs are unbounded logits, so they are squashed through a
sigmoid to land in [0, 1]. That keeps the score on the same scale the Engine's
``min_confidence`` gate expects.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..models import Hit
from ._util import require, sigmoid


class CrossEncoderReranker:
    """A Reranker backed by a sentence-transformers CrossEncoder.

    Args:
        model: HuggingFace cross-encoder model name.
        device: Torch device (e.g. "cpu", "cuda"); None lets the library decide.
        normalize: When True (default), map raw scores through a sigmoid to [0,1]
            so they stay compatible with the confidence gate.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        st = require("sentence_transformers", "rerank")
        self._model = st.CrossEncoder(model, device=device)
        self.normalize = normalize

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> List[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        rescored = [
            Hit(
                id=h.id,
                text=h.text,
                score=sigmoid(float(s)) if self.normalize else float(s),
                metadata=h.metadata,
            )
            for h, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]
