"""Tests for the Engine's post-draft verification layer.

`Engine(verify=True)` runs two checks over the drafted summary and attaches
their reports to the Answer: `grounding` (are the concrete values it asserted
actually in the evidence?) and `quality` (is the prose carried by that evidence,
or does it just restate the question?).

The properties pinned here are the ones a caller's trust rests on:

  - verification is OFF by default, so upgrading does not silently add an
    embedding call per answer;
  - it never changes the determination — it flags, and a human decides;
  - it does not run on a gated answer, and `None` reports mean "did not run",
    which must never be readable as a pass.

Deterministic fakes throughout — no network, no LLM.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

from attestq import Engine, Question, quality as aq

_DIM = 256


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_embed(texts: Sequence[str]) -> List[List[float]]:
    """Toy bag-of-words embedder: overlapping vocabulary -> high cosine."""
    vectors = []
    for text in texts:
        vec = [0.0] * _DIM
        for tok in _tokens(text):
            vec[hash(tok) % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


_EVIDENCE = (
    "Customer data at rest is encrypted using AES-256 or stronger. Encryption "
    "keys are rotated every 90 days and access to key material is restricted to "
    "named administrators under policy SEC-4."
)

_QUESTION = Question(
    id="ENC-1",
    prompt="Is customer data encrypted at rest under a managed key process?",
    choices=["Met", "Not Met", "Not Applicable"],
)


def _chat_returning(summary: str):
    def chat(prompt: str) -> str:
        return (
            "DETERMINATION: Met\n"
            f"EVIDENCE SUMMARY: {summary}\n"
            "CITATIONS: 1\n"
        )
    return chat


def _engine(summary: str, **kwargs) -> Engine:
    eng = Engine(chat=_chat_returning(summary), embed=keyword_embed,
                 min_confidence=0.0, **kwargs)
    eng.ingest([_EVIDENCE], namespace="v")
    return eng


# --------------------------------------------------------------- off by default

def test_verification_is_off_by_default():
    eng = _engine("Data at rest is encrypted using AES-256.")
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.grounding is None
    assert answer.quality is None
    assert answer.needs_review is False


def test_reports_are_attached_when_verify_is_on():
    eng = _engine("Data at rest is encrypted using AES-256.", verify=True)
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.grounding is not None
    assert answer.quality is not None


# --------------------------------------------------------------- grounding

def test_a_fabricated_specific_is_flagged():
    eng = _engine(
        "Data at rest is encrypted using AES-256 and the programme has been "
        "certified to ISO 27001 since 2019-03-01.",
        verify=True,
    )
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert "2019-03-01" in answer.grounding.unverified
    assert not answer.grounding.ok
    assert answer.needs_review


def test_a_specific_present_in_the_evidence_is_not_flagged():
    eng = _engine("Encryption keys are rotated every 90 days.", verify=True)
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert "90 days" in answer.grounding.checked
    assert answer.grounding.unverified == []


def test_verification_never_changes_the_determination():
    """The checks FLAG; disposition stays with the reviewer."""
    eng = _engine("Certified to ISO 27001 since 2019-03-01.", verify=True)
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.determination == "Met"
    assert answer.needs_review


# --------------------------------------------------------------- quality

def test_an_answer_lifted_from_the_evidence_is_supported():
    eng = _engine(
        "Customer data at rest is encrypted using AES-256 or stronger, and "
        "access to key material is restricted to named administrators.",
        verify=True,
    )
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.quality.verdict == aq.SUPPORTED
    assert not answer.quality.flagged


def test_a_pure_question_restatement_is_caught():
    eng = _engine(
        "Yes. Customer data is encrypted at rest under a managed key process.",
        verify=True,
    )
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.quality.flagged
    assert answer.needs_review


# --------------------------------------------------------------- the gated path

def test_a_gated_answer_is_not_verified():
    """No draft exists, so a clean report would be a lie of omission."""
    eng = Engine(chat=_chat_returning("unused"), embed=keyword_embed,
                 min_confidence=1.1, verify=True)
    eng.ingest([_EVIDENCE], namespace="v")
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.insufficient_evidence
    assert answer.grounding is None
    assert answer.quality is None
    assert answer.needs_review is False


# --------------------------------------------------------------- claim classifier

def test_the_claim_classifier_can_clear_a_flag():
    eng = _engine(
        "Yes. Customer data is encrypted at rest under a managed key process.",
        verify=True,
        claim_classifier=lambda q, a: False,   # "this item wants no evidence"
    )
    answer = eng.evaluate(_QUESTION, namespace="v")
    assert answer.quality.verdict == aq.NOT_APPLICABLE
    assert answer.quality.measured, "the pre-override verdict must be preserved"


def test_the_claim_classifier_is_ignored_when_verify_is_off():
    calls = []

    def classifier(q, a):
        calls.append(q)
        return False

    eng = _engine("Yes. Data is encrypted at rest.", claim_classifier=classifier)
    eng.evaluate(_QUESTION, namespace="v")
    assert calls == []


# --------------------------------------------------------------- serialization

def test_reports_survive_answers_to_dict():
    from attestq import answers_to_dict

    eng = _engine("Certified to ISO 27001 since 2019-03-01.", verify=True)
    answer = eng.evaluate(_QUESTION, namespace="v")
    d = answers_to_dict([answer])[0]
    assert "2019-03-01" in d["grounding"]["unverified"]
    assert d["quality"]["verdict"]


def test_reports_survive_to_json():
    import json

    from attestq import to_json

    eng = _engine("Certified to ISO 27001 since 2019-03-01.", verify=True)
    answer = eng.evaluate(_QUESTION, namespace="v")
    payload = json.loads(to_json([answer]))
    # Neither the standard nor the date is in this fixture's evidence, so both
    # are correctly unverified.
    assert payload["answers"][0]["grounding"]["unverified"] == ["ISO 27001", "2019-03-01"]
