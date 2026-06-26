"""Tests for the dependency-free core kernel.

Uses a deterministic keyword-overlap embedder and a scripted chat model so the
whole retrieve -> gate -> draft pipeline runs without any external services.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

import pytest

from attestq import (
    Answer,
    Engine,
    InMemoryVectorStore,
    Question,
    Questionnaire,
    cosine_similarity,
    parse_response,
    split_text,
)
from attestq.models import Hit

# --- deterministic fakes ------------------------------------------------------

_DIM = 256


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_embed(texts: Sequence[str]) -> List[List[float]]:
    """A toy bag-of-words embedder: overlapping vocabulary -> high cosine."""
    vectors = []
    for text in texts:
        vec = [0.0] * _DIM
        for tok in _tokens(text):
            vec[hash(tok) % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


class ScriptedChat:
    """Records prompts and returns a fixed labelled response."""

    def __init__(self, determination: str = "Met", citation: str = "1"):
        self.determination = determination
        self.citation = citation
        self.calls: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return (
            f"DETERMINATION: {self.determination}\n"
            f"EVIDENCE SUMMARY: The evidence supports this control.\n"
            f"CITATIONS: {self.citation}\n"
            f"NOTES: none"
        )


CHOICES = ["Met", "Not Met", "Not Applicable"]

ENCRYPTION_EVIDENCE = (
    "Data Protection Standard. All customer data at rest is encrypted using "
    "AES-256. Data in transit is encrypted using TLS 1.2 or higher. Cryptographic "
    "keys are managed in a dedicated key management service and rotated annually."
)
ACCESS_EVIDENCE = (
    "Access Control Standard. Multi-factor authentication is enforced for all "
    "remote and privileged access. Access is granted on a least-privilege basis "
    "and reviewed quarterly by system owners."
)


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def engine():
    chat = ScriptedChat()
    eng = Engine(chat=chat, embed=keyword_embed, store=InMemoryVectorStore())
    eng.chat_obj = chat  # keep a handle for assertions
    return eng


# --- tests --------------------------------------------------------------------


def test_ingest_returns_chunk_count(engine):
    n = engine.ingest([ENCRYPTION_EVIDENCE, ACCESS_EVIDENCE], namespace="acme")
    assert n == 2
    assert engine.store.count("acme") == 2


def test_evaluate_met_with_relevant_evidence(engine):
    engine.ingest([ENCRYPTION_EVIDENCE, ACCESS_EVIDENCE], namespace="acme")
    q = Question(id="ENC-1", prompt="Is customer data encrypted at rest?", choices=CHOICES)
    ans = engine.evaluate(q, namespace="acme")

    assert isinstance(ans, Answer)
    assert ans.determination == "Met"
    assert not ans.insufficient_evidence
    assert ans.confidence >= engine.min_confidence
    assert ans.citations, "expected at least one citation"
    assert "encrypt" in ans.citations[0].snippet.lower()
    assert engine.chat_obj.calls, "LLM should have been called"


def test_confidence_gate_fires_without_calling_llm(engine):
    engine.ingest([ENCRYPTION_EVIDENCE, ACCESS_EVIDENCE], namespace="acme")
    q = Question(
        id="GEO-1",
        prompt="What is the migratory pattern of arctic terns in winter?",
        choices=CHOICES,
    )
    ans = engine.evaluate(q, namespace="acme")

    assert ans.insufficient_evidence is True
    assert ans.determination == "Not Applicable"  # last choice = negative end
    assert ans.confidence < engine.min_confidence
    assert ans.citations == []
    assert engine.chat_obj.calls == [], "LLM must not be called when gate fires"


def test_insufficient_determination_without_choices(engine):
    engine.ingest([ENCRYPTION_EVIDENCE], namespace="acme")
    q = Question(id="X-1", prompt="totally unrelated quantum chromodynamics question")
    ans = engine.evaluate(q, namespace="acme")
    assert ans.insufficient_evidence is True
    assert ans.determination == "Insufficient Evidence"


def test_namespaces_are_isolated(engine):
    engine.ingest([ENCRYPTION_EVIDENCE], namespace="vendor-a")
    engine.ingest([ACCESS_EVIDENCE], namespace="vendor-b")
    assert engine.store.count("vendor-a") == 1
    assert engine.store.count("vendor-b") == 1

    q = Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=CHOICES)
    # vendor-b has no encryption evidence -> gate should fire there.
    ans_b = engine.evaluate(q, namespace="vendor-b")
    assert ans_b.insufficient_evidence is True


def test_evaluate_all_with_progress_callback(engine):
    engine.ingest([ENCRYPTION_EVIDENCE, ACCESS_EVIDENCE], namespace="acme")
    qn = Questionnaire(
        id="sec",
        title="Security Review",
        questions=[
            Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=CHOICES),
            Question(id="IAM-1", prompt="Is MFA enforced for privileged access?", choices=CHOICES),
        ],
    )
    seen = []
    answers = engine.evaluate_all(qn, namespace="acme", on_answer=seen.append)
    assert len(answers) == 2
    assert len(seen) == 2
    assert {a.question_id for a in answers} == {"ENC-1", "IAM-1"}


def test_ingest_accepts_metadata_forms(engine):
    engine.ingest(
        [
            (ENCRYPTION_EVIDENCE, {"source": "DataProtection.pdf"}),
            {"text": ACCESS_EVIDENCE, "filename": "AccessControl.docx", "tier": "2"},
        ],
        namespace="acme",
    )
    q = Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=CHOICES)
    ans = engine.evaluate(q, namespace="acme")
    assert ans.citations[0].source == "DataProtection.pdf"


# --- unit tests for helpers ---------------------------------------------------


def test_parse_response_tolerates_messy_output():
    hits = [Hit(id="c0", text="some evidence about encryption", score=0.9, metadata={"source": "doc.pdf"})]
    raw = (
        "DETERMINATION : Not Met\n"
        "EVIDENCE SUMMARY: The vendor did not provide retention periods.\n"
        "CITATIONS: 1\n"
        "NOTES: log retention unspecified"
    )
    determination, summary, citations = parse_response(raw, hits)
    assert determination == "Not Met"
    assert "retention" in summary.lower()
    assert "log retention unspecified" in summary.lower()
    assert citations[0].source == "doc.pdf"


def test_parse_response_missing_fields_falls_back():
    determination, summary, citations = parse_response("I think this is fine.", [])
    assert determination == "Undetermined"
    assert summary == "I think this is fine."
    assert citations == []


def test_split_text_overlaps_and_respects_size():
    text = ("Sentence one. " * 200).strip()
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)  # allow boundary slack


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
