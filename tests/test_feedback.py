"""Tests for the reviewer-correction scorecard (attestq/feedback.py).

This module turns "what the model drafted" vs "what the reviewer shipped" into
the only quality label a review workflow produces for free. Its job is to be
honest about small samples — the failure mode that matters is declaring the
confidence score miscalibrated off a handful of edits, which would be exactly
the overconfidence the module exists to detect.
"""

import pytest

from attestq import feedback as fb



def _outcome(draft="the drafted answer", final=None, confidence=None,
             sources=None, app="assurance", item_id="x", routed=False):
    return fb.DraftOutcome(
        item_id=item_id, app=app, prompt="q?", draft=draft, final=final,
        confidence=confidence, sources=sources or [], routed_to_sme=routed,
    )


# --------------------------------------------------------------- edit classification


def test_identical_answer_is_accepted():
    cls, ratio = fb.classify_edit("We encrypt data at rest.", "We encrypt data at rest.")
    assert cls == fb.ACCEPTED
    assert ratio == pytest.approx(1.0)


def test_whitespace_and_case_churn_is_not_an_edit():
    # Export/round-trip reformatting must not read as an analyst correction, or
    # the acceptance rate becomes a measure of the xlsx writer.
    cls, _ = fb.classify_edit("We encrypt data\nat rest.", "  we encrypt data at rest.  ")
    assert cls == fb.ACCEPTED


def test_tightened_wording_is_a_light_edit():
    cls, _ = fb.classify_edit(
        "Yes, we do encrypt all customer data at rest using AES-256 encryption.",
        "Yes, we encrypt all customer data at rest using AES-256.",
    )
    assert cls == fb.LIGHT_EDIT


def test_a_wholly_different_answer_is_replaced():
    cls, _ = fb.classify_edit(
        "We do not currently hold a SOC 2 Type II report.",
        "Acme Corporation maintains an ISO 27001 certification renewed annually by an "
        "accredited external auditor; evidence available under NDA.",
    )
    assert cls == fb.REPLACED


def test_missing_final_is_unreviewed_not_rejected():
    # Work in flight is not a failed draft. Counting it as one would make the
    # acceptance rate a function of queue depth.
    assert fb.classify_edit("a draft", None)[0] == fb.UNREVIEWED
    assert fb.classify_edit("a draft", "   ")[0] == fb.UNREVIEWED
    assert _outcome(final=None).reviewed is False


def test_empty_draft_with_a_real_final_is_replaced():
    cls, _ = fb.classify_edit("", "The analyst wrote this from scratch.")
    assert cls == fb.REPLACED


# --------------------------------------------------------------- calibration


def _banded(pairs):
    """pairs of (confidence, accepted?) -> outcomes"""
    out = []
    for i, (conf, ok) in enumerate(pairs):
        draft = "identical text"
        final = "identical text" if ok else "a completely different response entirely"
        out.append(_outcome(draft=draft, final=final, confidence=conf, item_id=str(i)))
    return out


def test_confidence_bands_bucket_and_count():
    stats = fb.calibration_by_confidence(
        _banded([(0.20, False), (0.60, True), (0.61, False), (0.95, True)])
    )
    by_label = {s.label: s for s in stats}
    assert by_label["0.00–0.55"].total == 1
    assert by_label["0.55–0.72"].total == 2
    assert by_label["0.55–0.72"].accepted == 1
    assert by_label["0.88–1.01"].accepted == 1


def test_outcomes_without_confidence_are_skipped_not_zeroed():
    # Bucketing a null confidence at 0.0 would manufacture a fake
    # low-confidence failure population and slander the gate.
    stats = fb.calibration_by_confidence([_outcome(final="identical", draft="identical")])
    assert sum(s.total for s in stats) == 0


def test_unreviewed_outcomes_never_enter_calibration():
    stats = fb.calibration_by_confidence([_outcome(final=None, confidence=0.9)])
    assert sum(s.total for s in stats) == 0


def test_calibration_verdict_withheld_on_thin_data():
    # Seven items per band is exactly the volume where one edit swings a band
    # 15 points. The honest answer is "don't know".
    stats = fb.calibration_by_confidence(
        _banded([(0.20, False)] * 7 + [(0.60, True)] * 5 + [(0.80, True)] * 7)
    )
    assert fb.is_calibrated(stats) is None


def test_monotonic_acceptance_reads_as_calibrated():
    stats = fb.calibration_by_confidence(
        _banded([(0.20, False)] * 12 + [(0.60, True)] * 6 + [(0.60, False)] * 6
                + [(0.95, True)] * 12)
    )
    assert fb.is_calibrated(stats) is True


def test_inverted_acceptance_reads_as_miscalibrated():
    stats = fb.calibration_by_confidence(
        _banded([(0.20, True)] * 12 + [(0.95, False)] * 12)
    )
    assert fb.is_calibrated(stats) is False


def test_gate_check_pools_each_side_of_the_threshold():
    # The robust cut: it can say something useful long before any single band
    # has the samples to be trusted.
    g = fb.gate_check(_banded([(0.30, False)] * 7 + [(0.70, True)] * 9 + [(0.70, False)] * 3),
                      gate=0.55)
    assert g["below_n"] == 7
    assert g["below_acceptance"] == pytest.approx(0.0)
    assert g["above_n"] == 12
    assert g["above_acceptance"] == pytest.approx(0.75)


# --------------------------------------------------------------- source trust


def test_source_trust_ranks_worst_first_and_needs_repeat_use():
    good = [_outcome(draft="same", final="same", sources=["good.docx"], item_id=f"g{i}")
            for i in range(4)]
    bad = [_outcome(draft="same", final="utterly different prose here",
                    sources=["bad.xlsx"], item_id=f"b{i}") for i in range(4)]
    once = [_outcome(draft="same", final="different", sources=["rare.pdf"])]

    ranked = fb.source_trust(good + bad + once, min_uses=3)
    names = [s.source for s in ranked]

    assert names == ["bad.xlsx", "good.docx"]        # least trusted first
    assert "rare.pdf" not in names                    # one use is an anecdote
    assert ranked[0].trust == pytest.approx(0.0)
    assert ranked[1].trust == pytest.approx(1.0)


def test_a_source_cited_twice_in_one_answer_counts_once():
    o = _outcome(draft="same", final="same", sources=["dup.docx", "dup.docx", "dup.docx"])
    ranked = fb.source_trust([o] * 3, min_uses=3)
    assert ranked[0].total == 3


# --------------------------------------------------------------- scorecard


def test_scorecard_reports_the_sme_routing_ceiling():
    # Every SME-routed question is one a human answers from scratch — the
    # throughput number a daily user actually feels.
    outcomes = ([_outcome(final=None, routed=True) for _ in range(3)]
                + [_outcome(draft="same", final="same")])
    card = fb.build_scorecard(outcomes)

    assert card["totals"]["items"] == 4
    assert card["totals"]["routed_to_sme"] == 3
    assert card["totals"]["sme_routing_rate"] == pytest.approx(0.75)
    assert card["totals"]["reviewed"] == 1


def test_scorecard_rates_are_over_reviewed_items_only():
    outcomes = [_outcome(draft="same", final="same"),
                _outcome(draft="same", final="totally other text entirely"),
                _outcome(final=None), _outcome(final=None)]
    card = fb.build_scorecard(outcomes)

    # 1 of 2 REVIEWED, not 1 of 4 total.
    assert card["acceptance_rate"] == pytest.approx(0.5)
    assert card["totals"]["unreviewed"] == 2


def test_scorecard_splits_by_app():
    outcomes = [_outcome(draft="same", final="same", app="assurance"),
                _outcome(draft="same", final="other entirely different", app="vendor-risk")]
    card = fb.build_scorecard(outcomes)

    assert card["by_app"]["assurance"]["acceptance_rate"] == pytest.approx(1.0)
    assert card["by_app"]["vendor-risk"]["acceptance_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------- Answer adapter
# `outcome_from_answer` is the bridge from a drafted attestq Answer to a record
# this module can score. It is duck-typed on purpose, so these cover both a real
# Engine answer and a bare stand-in.


def _engine_answer(**engine_kwargs):
    """Draft one real answer through the kernel, with deterministic fakes."""
    import math
    import re

    from attestq import Engine, Question

    def embed(texts):
        vectors = []
        for text in texts:
            vec = [0.0] * 128
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[hash(tok) % 128] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def chat(_prompt):
        return ("DETERMINATION: Met\n"
                "EVIDENCE SUMMARY: Data at rest is encrypted using AES-256.\n"
                "CITATIONS: 1\n")

    engine_kwargs.setdefault("min_confidence", 0.0)
    eng = Engine(chat=chat, embed=embed, **engine_kwargs)
    eng.ingest(["Customer data at rest is encrypted using AES-256 or stronger."],
               namespace="v")
    q = Question(id="ENC-1", prompt="Is data encrypted at rest?",
                 choices=["Met", "Not Met"])
    return eng.evaluate(q, namespace="v"), q


def test_an_accepted_engine_answer_round_trips():
    answer, question = _engine_answer()
    outcome = fb.outcome_from_answer(answer, question, final=answer.summary)

    assert outcome.item_id == "ENC-1"
    assert outcome.prompt == "Is data encrypted at rest?"
    assert outcome.edit_class == fb.ACCEPTED
    assert outcome.confidence == answer.confidence
    assert outcome.sources, "citations should populate sources"


def test_an_edited_engine_answer_is_not_accepted():
    answer, question = _engine_answer()
    outcome = fb.outcome_from_answer(
        answer, question,
        final="Data at rest uses AES-256, and keys rotate every 90 days per SEC-4.",
    )
    assert outcome.edit_class != fb.ACCEPTED
    assert outcome.reviewed


def test_an_unreviewed_answer_carries_no_signal():
    """final=None is work in flight, not a rejection."""
    answer, question = _engine_answer()
    outcome = fb.outcome_from_answer(answer, question, final=None)
    assert outcome.edit_class == fb.UNREVIEWED
    assert not outcome.reviewed
    assert fb.build_scorecard([outcome])["acceptance_rate"] is None


def test_a_gated_answer_counts_as_routed_to_an_sme():
    """The gate firing means a human answers it from scratch."""
    answer, question = _engine_answer(min_confidence=1.1)
    assert answer.insufficient_evidence
    outcome = fb.outcome_from_answer(answer, question, final=None)
    assert outcome.routed_to_sme
    assert fb.build_scorecard([outcome])["totals"]["sme_routing_rate"] == 1.0


def test_citation_snippets_and_scores_are_carried_over():
    answer, question = _engine_answer()
    outcome = fb.outcome_from_answer(answer, question, final=answer.summary)
    assert outcome.citations
    assert outcome.citations[0].text, "snippet should map onto Citation.text"
    assert outcome.top_citation_score is not None


def test_it_is_duck_typed_not_bound_to_the_answer_class():
    """A record from a system that never touched attestq still folds in."""
    class _Foreign:
        question_id = "EXT-1"
        summary = "Drafted elsewhere entirely."
        citations = []
        confidence = 0.81
        insufficient_evidence = False

    outcome = fb.outcome_from_answer(_Foreign(), None, final="Drafted elsewhere entirely.",
                                     app="legacy")
    assert outcome.app == "legacy"
    assert outcome.edit_class == fb.ACCEPTED
    assert outcome.prompt == ""


def test_outcomes_from_answers_aggregate_into_a_scorecard():
    answer, question = _engine_answer()
    outcomes = [
        fb.outcome_from_answer(answer, question, final=answer.summary),
        fb.outcome_from_answer(answer, question, final="Something entirely different."),
    ]
    card = fb.build_scorecard(outcomes)
    assert card["totals"]["reviewed"] == 2
    assert card["acceptance_rate"] == 0.5
    assert card["by_app"]["attestq"]["reviewed"] == 2
