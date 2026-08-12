"""Tests for the LLM claim classifier (attestq/claim_classifier.py).

The classifier answers one question — "should a good answer to this be backed by
documentation?" — so `quality.assess` can drop a flag it raised on an item that
never wanted evidence ("What is the legal entity name?").

Two properties are load-bearing, and most of these tests exist to pin them:

  1. **It fails closed.** Every failure path — unreachable model, raised
     exception, prose instead of a word, empty reply — must return True
     (evidence required), which leaves the verdict exactly as measurement found
     it. A classifier that failed *open* would silently clear real flags.

  2. **It classifies the question, not the answer.** The answer is accepted to
     match the `ClaimClassifier` signature and must never reach the model, or
     the classifier starts rationalizing from what was written.

No network: a fake chat callable stands in for the model everywhere.
"""

import pytest

from attestq.claim_classifier import (
    LLMClaimClassifier,
    make_claim_classifier,
    parse_reply,
)


class _FakeChat:
    """A `ChatFn` that returns a canned reply, or raises if given an exception.

    Records every prompt it was handed so tests can assert on what was sent.
    """

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _classifier(reply, **kwargs):
    chat = _FakeChat(reply)
    return make_claim_classifier(chat, **kwargs), chat


# --------------------------------------------------------------- the happy path

def test_an_evidence_question_needs_evidence():
    clf, _ = _classifier("EVIDENCE")
    assert clf("Are encryption keys managed under a documented process?") is True


def test_a_record_question_does_not():
    clf, _ = _classifier("RECORD")
    assert clf("What is your registered legal entity name?") is False


def test_the_question_is_what_gets_classified_not_the_answer():
    clf, chat = _classifier("RECORD")
    clf("What is your legal entity name?", "Acme Corp is certified to ISO 27001.")
    assert len(chat.prompts) == 1
    sent = chat.prompts[0]
    assert "What is your legal entity name?" in sent
    # The answer must not reach the model — see property 2 in the module docstring.
    assert "Acme Corp" not in sent
    assert "ISO 27001" not in sent


# --------------------------------------------------------------- fails closed

def test_a_raising_model_keeps_the_flag():
    clf, _ = _classifier(ConnectionError("inference endpoint down"))
    assert clf("Are backups tested?") is True


def test_prose_instead_of_a_word_keeps_the_flag():
    clf, _ = _classifier(
        "This item is asking about the organization's controls, so I would say "
        "it is an EVIDENCE item rather than a RECORD item."
    )
    assert clf("Are backups tested?") is True


def test_an_empty_reply_keeps_the_flag():
    clf, _ = _classifier("")
    assert clf("Are backups tested?") is True


def test_an_empty_question_is_not_sent_to_the_model():
    clf, chat = _classifier("RECORD")
    assert clf("") is True
    assert clf("   ") is True
    assert chat.prompts == []


# --------------------------------------------------------------- reply parsing

def test_a_reasoning_block_is_stripped_before_parsing():
    clf, _ = _classifier(
        "<think>The item asks for a fact of record about the entity, not a "
        "control. So this is a RECORD item.</think>RECORD"
    )
    assert clf("What is your DUNS number?") is False


def test_decoration_around_the_word_is_tolerated():
    clf, _ = _classifier("**RECORD**")
    assert clf("What is your headquarters address?") is False


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("EVIDENCE", True),
        ("RECORD", False),
        ("evidence", True),
        ("record.", False),
        ("EVIDENCE - it asks about a control", True),
        ("", None),
        ("I am not sure.", None),
        ("Both EVIDENCE and RECORD apply here.", None),
    ],
)
def test_parse_reply_is_strict_about_the_leading_word(reply, expected):
    """Unparseable resolves to None so the caller keeps the flag AND declines to
    cache — a paragraph mentioning both words must not resolve on token order."""
    assert parse_reply(reply) is expected


# --------------------------------------------------------------- caching

def test_a_repeated_question_is_answered_from_cache():
    clf, chat = _classifier("RECORD")
    q = "What is your registered legal entity name?"
    assert clf(q) is False
    assert clf(q) is False
    assert len(chat.prompts) == 1, "second call should not have hit the model"


def test_the_cache_key_ignores_case_and_whitespace():
    clf, chat = _classifier("RECORD")
    assert clf("What is your legal entity name?") is False
    assert clf("  WHAT   IS your Legal   Entity NAME?  ") is False
    assert len(chat.prompts) == 1


def test_an_unparseable_reply_is_not_cached():
    """A model problem must not be frozen into a question property."""
    clf, chat = _classifier("I am not sure.")
    q = "Are backups tested?"
    assert clf(q) is True
    assert clf(q) is True
    assert len(chat.prompts) == 2, "an unparseable reply should be retried"


def test_reset_cache_forces_reclassification():
    clf, chat = _classifier("RECORD")
    q = "What is your legal entity name?"
    clf(q)
    assert clf.cache_size == 1
    clf.reset_cache()
    assert clf.cache_size == 0
    clf(q)
    assert len(chat.prompts) == 2


def test_the_cache_is_bounded():
    clf, _ = _classifier("RECORD", cache_limit=3)
    for i in range(5):
        clf(f"What is registered identifier number {i}?")
    assert clf.cache_size <= 3


# --------------------------------------------------------------- wiring

def test_it_satisfies_the_claim_classifier_contract_assess_expects():
    """`quality.assess` calls it as (question, answer) and may only downgrade."""
    from attestq import quality as aq

    question = "What is your registered legal entity name?"
    answer = "Our registered legal entity name is Acme Corporation."
    evidence = "Encryption keys are rotated every 90 days under policy SEC-4."

    flagged = aq.assess(answer, context=evidence, question=question)
    assert flagged.flagged, "an entity-name answer is unsupported by control evidence"

    clf, _ = _classifier("RECORD")
    cleared = aq.assess(answer, context=evidence, question=question,
                        claim_classifier=clf)
    assert cleared.verdict == aq.NOT_APPLICABLE
    assert cleared.measured == flagged.verdict, "the override must stay auditable"


def test_a_classifier_that_says_evidence_leaves_the_flag_standing():
    from attestq import quality as aq

    question = "Are encryption keys rotated on a defined schedule?"
    answer = "Yes, encryption keys are rotated on a defined schedule."
    evidence = "The vendor operates a 24x7 security operations centre."

    clf, _ = _classifier("EVIDENCE")
    report = aq.assess(answer, context=evidence, question=question,
                       claim_classifier=clf)
    assert report.flagged
    assert report.verdict != aq.NOT_APPLICABLE


def test_the_class_and_the_factory_agree():
    chat = _FakeChat("RECORD")
    direct = LLMClaimClassifier(chat)
    assert direct("What is your legal entity name?") is False
