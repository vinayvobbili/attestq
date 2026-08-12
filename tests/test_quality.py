"""Tests for the referenceless answer-quality checks (attestq/quality.py).

The failure this module exists to catch is an answer that restates its question
and cites evidence it never drew on — invisible to grounding.py, which only
verifies concrete values and therefore passes a generic "Yes, we do X"
vacuously.

Two properties are load-bearing:

  1. No absolute similarity constant. Support is judged against the question's
     OWN similarity to the evidence, so the checks survive an embedding-model
     swap. The tests below assert on relationships, never on a magic number.

  2. Fail-open. A verification pass that breaks drafting is worse than no
     verification pass, so a dead embedding endpoint must degrade to lexical
     scoring and an absent signal must never read as a pass.
"""

import math

from attestq import quality as aq

_QUESTION = ("Are encryption keys securely managed, restricted, and controlled "
             "through defined key-management processes?")

# The kind of accepted answer that motivated the module: the
# first sentence is the question with "Yes." in front, the second is genuinely
# lifted from the policy document beside it.
_ECHO_SENTENCE = ("Encryption keys are securely managed, restricted, and "
                  "controlled through defined key-management processes.")
_GROUNDED_SENTENCE = ("Customer data at rest is encrypted using AES-256 or "
                      "stronger, with access to keys limited to authorized "
                      "personnel in accordance with enterprise policy.")
_EVIDENCE = ("Customer data at rest must be encrypted using AES-256 or stronger. "
             "Data in transit must use TLS 1.2 or higher. Deprecated protocols "
             "including SSL 3.0 must be disabled on all production endpoints.")


def _bag_embed(texts):
    """Deterministic stand-in for the embedding endpoint.

    Term-frequency vectors over the union vocabulary. Crude, but it has the one
    property the tests need: texts that share wording land close together, so
    cosine behaves qualitatively like the real embedder without a network call.
    """
    tokenized = [t.lower().replace(".", " ").replace(",", " ").split() for t in texts]
    vocab = sorted({w for toks in tokenized for w in toks})
    index = {w: i for i, w in enumerate(vocab)}
    vectors = []
    for toks in tokenized:
        vec = [0.0] * len(vocab)
        for w in toks:
            vec[index[w]] += 1.0
        vectors.append(vec)
    return vectors


# --------------------------------------------------------------- primitives


def test_sentences_split_on_terminal_punctuation_and_newlines():
    assert aq.sentences("One claim. Two claims!\nThree?") == [
        "One claim.", "Two claims!", "Three?"]


def test_sentences_drop_inline_citation_markers():
    # "[Source: P1]" is provenance, not an assertion. Left in, every citing
    # answer gets docked for a word that appears in no evidence.
    out = aq.sentences("Keys are rotated annually [Source: P1].")
    assert "Source" not in out[0] and "P1" not in out[0]
    assert "rotated annually" in out[0]


def test_cosine_of_orthogonal_and_identical_vectors():
    assert aq.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert math.isclose(aq.cosine([1.0, 2.0], [1.0, 2.0]), 1.0, rel_tol=1e-9)
    assert aq.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0    # no division by zero


# --------------------------------------------------------------- the core signal


def test_question_restatement_scores_as_echo_not_support():
    """The whole point: a sentence nearer the question than the evidence."""
    scores = aq._embedded_scores(
        [_ECHO_SENTENCE, _GROUNDED_SENTENCE], _QUESTION, [_EVIDENCE], _bag_embed)
    echoed, grounded = scores

    assert echoed.is_echo and not echoed.is_supported
    assert grounded.is_supported and not grounded.is_echo


def test_answer_that_only_restates_the_question_is_vacuous():
    report = aq.assess(f"Yes. {_ECHO_SENTENCE}", context=_EVIDENCE,
                       question=_QUESTION, embed_fn=_bag_embed)
    assert report.verdict == aq.VACUOUS
    assert report.flagged
    assert report.echo == 1.0


def test_answer_drawn_from_evidence_is_supported():
    report = aq.assess(_GROUNDED_SENTENCE, context=_EVIDENCE,
                       question=_QUESTION, embed_fn=_bag_embed)
    assert report.verdict == aq.SUPPORTED
    assert not report.flagged


def test_mixed_answer_reports_the_unsupported_sentence():
    report = aq.assess(f"{_ECHO_SENTENCE} {_GROUNDED_SENTENCE}", context=_EVIDENCE,
                       question=_QUESTION, embed_fn=_bag_embed)
    assert report.support == 0.5
    assert report.unsupported == [_ECHO_SENTENCE]


def test_support_is_judged_against_the_question_baseline_not_a_constant():
    """Identical prose, two evidence sets: the verdict must follow the baseline.

    This is what a hardcoded cosine floor cannot do, and why one is not in the
    module. Against evidence the question itself matches closely, a sentence has
    to clear a higher bar than against evidence it barely matches.
    """
    sentence = "Keys are rotated annually inside a hardware security module."

    # Evidence the question barely resembles: a low bar, and the sentence clears
    # it because it genuinely came from there.
    distant = "Keys are rotated annually inside a hardware security module."
    against_distant = aq._embedded_scores([sentence], _QUESTION, [distant], _bag_embed)[0]

    # Evidence that is largely the question restated: a high bar, and the same
    # sentence no longer clears it — it is not adding what that evidence holds.
    near = _QUESTION + " Encryption keys are managed through defined processes."
    against_near = aq._embedded_scores([sentence], _QUESTION, [near], _bag_embed)[0]

    assert against_near.baseline > against_distant.baseline
    assert against_distant.is_supported
    assert not against_near.is_supported

    # The baseline is exactly the question's own similarity to that evidence —
    # a real per-question quantity, not a leftover default.
    q_vec, ev_vec = _bag_embed([_QUESTION, distant])
    assert math.isclose(against_distant.baseline, aq.cosine(q_vec, ev_vec), rel_tol=1e-9)


# --------------------------------------------------------------- absent signal


def test_no_evidence_is_reported_rather_than_passed():
    report = aq.assess(_GROUNDED_SENTENCE, context="", question=_QUESTION,
                       embed_fn=_bag_embed)
    assert report.verdict == aq.NO_EVIDENCE
    assert not report.has_evidence
    # Not flagged — an empty corpus is the retrieval gate's problem to report,
    # not this check's — but the verdict must say so rather than read "fine".
    assert report.ok and report.verdict != aq.SUPPORTED


def test_empty_answer_is_unscored():
    assert aq.assess("", context=_EVIDENCE, question=_QUESTION).verdict == aq.UNSCORED


def test_answer_of_only_fragments_is_unscored():
    # "United States" answers a question honestly and has no claim to check.
    report = aq.assess("United States", context=_EVIDENCE, question=_QUESTION,
                       embed_fn=_bag_embed)
    assert report.verdict == aq.UNSCORED
    assert not report.flagged


def test_a_dead_embedding_endpoint_falls_back_to_lexical():
    def broken(_texts):
        raise ConnectionError("both Macs down")

    report = aq.assess(_GROUNDED_SENTENCE, context=_EVIDENCE, question=_QUESTION,
                       embed_fn=broken)
    assert report.method == "lexical"
    assert report.verdict != aq.UNSCORED


def test_lexical_path_still_separates_echo_from_grounded():
    report = aq.assess(f"Yes. {_ECHO_SENTENCE}", context=_EVIDENCE, question=_QUESTION)
    assert report.method == "lexical"
    assert report.verdict == aq.VACUOUS


# --------------------------------------------------------------- claim classifier


def test_claim_classifier_can_only_downgrade_a_flag():
    admin_q = "What is the company/business name?"
    answer = "The company business name is Acme Corporation and it is a listed insurer."

    flagged = aq.assess(answer, context=_EVIDENCE, question=admin_q, embed_fn=_bag_embed)
    assert flagged.flagged

    cleared = aq.assess(answer, context=_EVIDENCE, question=admin_q,
                        embed_fn=_bag_embed, claim_classifier=lambda q, a: False)
    assert cleared.verdict == aq.NOT_APPLICABLE
    assert not cleared.flagged


def test_a_downgrade_preserves_what_measurement_concluded():
    """The override has to stay auditable.

    NOT_APPLICABLE on its own cannot tell "correctly cleared a company-name
    lookup" from "cleared a vacuous answer to a real control question", and a
    downgrade writes no marker, so this field is the only surviving evidence
    that the classifier overrode anything.
    """
    admin_q = "What is the company/business name?"
    answer = "The company business name is Acme Corporation and it is a listed insurer."

    flagged = aq.assess(answer, context=_EVIDENCE, question=admin_q, embed_fn=_bag_embed)
    cleared = aq.assess(answer, context=_EVIDENCE, question=admin_q,
                        embed_fn=_bag_embed, claim_classifier=lambda q, a: False)

    assert cleared.measured == flagged.verdict
    assert cleared.measured in aq.FLAGGED_VERDICTS
    assert flagged.verdict in cleared.detail() and "%" in cleared.detail()
    assert cleared.as_dict()["measured"] == flagged.verdict


def test_no_downgrade_leaves_the_measured_field_empty():
    # Only a downgrade populates it, so a non-empty value always means an
    # override happened — no need to also check the verdict to interpret it.
    report = aq.assess(_GROUNDED_SENTENCE, context=_EVIDENCE, question=_QUESTION,
                       embed_fn=_bag_embed)
    assert report.measured == ""
    assert "measured" not in report.detail()


def test_claim_classifier_cannot_create_a_flag():
    # A classifier insisting everything needs evidence must not turn a supported
    # answer into a flagged one — it is consulted only on already-flagged items.
    report = aq.assess(_GROUNDED_SENTENCE, context=_EVIDENCE, question=_QUESTION,
                       embed_fn=_bag_embed, claim_classifier=lambda q, a: True)
    assert report.verdict == aq.SUPPORTED


def test_a_raising_claim_classifier_leaves_the_verdict_alone():
    def broken(_q, _a):
        raise RuntimeError("classifier endpoint down")

    report = aq.assess(f"Yes. {_ECHO_SENTENCE}", context=_EVIDENCE, question=_QUESTION,
                       embed_fn=_bag_embed, claim_classifier=broken)
    assert report.verdict == aq.VACUOUS


# --------------------------------------------------------------- marker round-trip


def test_annotate_is_a_no_op_for_a_clean_report():
    clean = aq.assess(_GROUNDED_SENTENCE, context=_EVIDENCE, question=_QUESTION,
                      embed_fn=_bag_embed)
    assert aq.annotate("original rationale", clean) == "original rationale"


def test_annotate_then_parse_and_strip_round_trip():
    report = aq.assess(f"Yes. {_ECHO_SENTENCE}", context=_EVIDENCE,
                       question=_QUESTION, embed_fn=_bag_embed)
    stamped = aq.annotate("Drafted from the encryption policy.", report)

    assert aq.has_weak_support(stamped)
    assert aq.parse_marker(stamped) == report.detail()
    assert aq.strip_marker(stamped) == "Drafted from the encryption policy."


def test_marker_read_back_ignores_unmarked_text():
    assert not aq.has_weak_support("plain rationale")
    assert aq.parse_marker("plain rationale") == ""
    assert aq.strip_marker("plain rationale") == "plain rationale"


# --------------------------------------------------------------- offline helper


def test_evidence_support_scores_a_single_passage():
    assert aq.evidence_support(_GROUNDED_SENTENCE, _EVIDENCE, _QUESTION) > 0.0


def test_evidence_support_rejects_a_question_restatement():
    # The filter that keeps echo answers out of the reranker's positive pairs.
    assert aq.evidence_support(_ECHO_SENTENCE, _EVIDENCE, _QUESTION) == 0.0


def test_evidence_support_rejects_an_unrelated_passage():
    assert aq.evidence_support(
        _GROUNDED_SENTENCE,
        "Banana bread is best baked at 175 degrees for fifty minutes.",
        _QUESTION,
    ) == 0.0
