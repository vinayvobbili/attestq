"""Tests for 0.2.0 gating controls: gate_on and insufficient-evidence overrides."""

from __future__ import annotations

import math
import re
from typing import List, Sequence

import pytest

from attestq import Engine, InMemoryVectorStore, Question
from attestq.models import Hit

_DIM = 256


def keyword_embed(texts: Sequence[str]) -> List[List[float]]:
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            v[hash(tok) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def scripted_chat(prompt: str) -> str:
    return "DETERMINATION: Met\nEVIDENCE SUMMARY: ok\nCITATIONS: 1\nNOTES: none"


CHOICES = ["Met", "Not Met", "Not Applicable"]
EVIDENCE = "All customer data at rest is encrypted using AES-256 and TLS 1.2 in transit."


class _ConstantReranker:
    """Reranker that overwrites every hit's score with a fixed low value.

    Lets us prove gate_on='retrieval' ignores the (low) rerank score while
    gate_on='rerank' is driven by it.
    """

    def __init__(self, score: float):
        self.score = score

    def rerank(self, query, hits: Sequence[Hit], top_k: int) -> List[Hit]:
        return [Hit(id=h.id, text=h.text, score=self.score, metadata=h.metadata) for h in hits][:top_k]


def _engine(**kw):
    return Engine(chat=scripted_chat, embed=keyword_embed, store=InMemoryVectorStore(), **kw)


def test_gate_on_retrieval_ignores_low_rerank_score():
    eng = _engine(reranker=_ConstantReranker(0.01), gate_on="retrieval", min_confidence=0.45)
    eng.ingest([EVIDENCE], namespace="v")
    ans = eng.evaluate(Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=CHOICES), namespace="v")
    # Strong retrieval match -> passes the gate even though rerank score is 0.01.
    assert not ans.insufficient_evidence
    assert ans.determination == "Met"
    assert ans.confidence >= 0.45


def test_gate_on_rerank_uses_post_rerank_score():
    eng = _engine(reranker=_ConstantReranker(0.01), gate_on="rerank", min_confidence=0.45)
    eng.ingest([EVIDENCE], namespace="v")
    ans = eng.evaluate(Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=CHOICES), namespace="v")
    # Now the low rerank score drives the gate -> insufficient.
    assert ans.insufficient_evidence
    assert math.isclose(ans.confidence, 0.01, rel_tol=1e-6)


def test_default_gate_on_is_retrieval():
    assert _engine().gate_on == "retrieval"


def test_invalid_gate_on_rejected():
    with pytest.raises(ValueError):
        _engine(gate_on="bogus")


def test_insufficient_determination_override():
    # A vendor-risk case: gate outcome must be "Not Met", not the last choice
    # "Not Applicable" — absent evidence is a gap, not an exemption.
    eng = _engine(insufficient_determination="Not Met",
                  insufficient_summary="Not provided by vendor documentation.")
    eng.ingest([EVIDENCE], namespace="v")
    ans = eng.evaluate(
        Question(id="GAP-1", prompt="totally unrelated migratory bird question", choices=CHOICES),
        namespace="v",
    )
    assert ans.insufficient_evidence
    assert ans.determination == "Not Met"  # override, not choices[-1] ("Not Applicable")
    assert ans.summary == "Not provided by vendor documentation."


def test_insufficient_default_still_uses_last_choice():
    eng = _engine()
    eng.ingest([EVIDENCE], namespace="v")
    ans = eng.evaluate(
        Question(id="GAP-1", prompt="totally unrelated migratory bird question", choices=CHOICES),
        namespace="v",
    )
    assert ans.insufficient_evidence
    assert ans.determination == "Not Applicable"  # choices[-1] fallback unchanged
