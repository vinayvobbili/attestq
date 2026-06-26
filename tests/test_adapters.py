"""Tests for optional adapters.

The kernel must import and behave sanely whether or not the heavy extras are
installed. These tests cover the dep-free helpers, the friendly-error contract
when an extra is missing, and a real end-to-end roundtrip with Chroma when it
happens to be available.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from attestq.adapters._util import require, sigmoid


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --- helpers ------------------------------------------------------------------


def test_require_returns_installed_module():
    mod = require("math", "irrelevant")
    assert mod is math


def test_require_raises_friendly_hint_for_missing_module():
    with pytest.raises(ImportError) as exc:
        require("definitely_not_a_real_module_xyz", "chroma")
    assert 'attestq[chroma]' in str(exc.value)


def test_sigmoid_bounds_and_monotonic():
    assert 0.0 < sigmoid(-10) < sigmoid(0) < sigmoid(10) < 1.0
    assert sigmoid(0) == pytest.approx(0.5)
    assert 0.0 <= sigmoid(-50) and sigmoid(50) <= 1.0  # saturates, stays in range


# --- missing-extra contract ---------------------------------------------------


@pytest.mark.skipif(_installed("openai"), reason="openai is installed")
def test_openai_adapter_errors_without_extra():
    from attestq.adapters import OpenAIChat

    with pytest.raises(ImportError) as exc:
        OpenAIChat(model="gpt-4o-mini")
    assert "attestq[openai]" in str(exc.value)


@pytest.mark.skipif(_installed("chromadb"), reason="chromadb is installed")
def test_chroma_adapter_errors_without_extra():
    from attestq.adapters import ChromaStore

    with pytest.raises(ImportError) as exc:
        ChromaStore()
    assert "attestq[chroma]" in str(exc.value)


@pytest.mark.skipif(_installed("sentence_transformers"), reason="sentence-transformers installed")
def test_reranker_errors_without_extra():
    from attestq.adapters import CrossEncoderReranker

    with pytest.raises(ImportError) as exc:
        CrossEncoderReranker()
    assert "attestq[rerank]" in str(exc.value)


# --- real Chroma roundtrip (only when installed) ------------------------------


@pytest.mark.skipif(not _installed("chromadb"), reason="chromadb not installed")
def test_chroma_store_roundtrip_via_engine():
    import re

    from attestq import Engine, Question
    from attestq.adapters import ChromaStore

    _DIM = 256

    def keyword_embed(texts):
        vecs = []
        for t in texts:
            v = [0.0] * _DIM
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[hash(tok) % _DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs

    def chat(prompt):
        return "DETERMINATION: Met\nEVIDENCE SUMMARY: ok\nCITATIONS: 1\nNOTES: none"

    eng = Engine(chat=chat, embed=keyword_embed, store=ChromaStore())
    eng.ingest(["All data at rest is encrypted with AES-256."], namespace="v1")
    assert eng.store.count("v1") == 1
    ans = eng.evaluate(
        Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=["Met", "Not Met"]),
        namespace="v1",
    )
    assert ans.determination == "Met"
    assert ans.citations
