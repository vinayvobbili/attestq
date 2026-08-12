"""Tests for the deterministic answer-grounding check (attestq/grounding.py).

This is the trust-but-verify layer for drafted answers: after the model drafts,
we confirm every concrete value it asserted (date, version, percentage,
standard, duration) actually occurs in the evidence it was drafted from. A value
that doesn't is an "unverified specific" — the highest-liability hallucination
for a compliance answer someone will act on.

The module is pure stdlib with no intra-package imports, and these tests need no
DB, network, or LLM — the same isolation the verifier itself relies on.
"""

from attestq import grounding



# --------------------------------------------------------------- extract_specifics

def test_extract_specifics_catches_the_high_liability_value_classes():
    text = (
        "Certified to ISO 27001 and SOC 2 since 2019-03-01, data encrypted with "
        "AES-256 over TLS 1.2, keys rotated every 90 days, 99.9% uptime, on policy v4.2."
    )
    found = {s.lower() for s in grounding.extract_specifics(text)}
    assert "2019-03-01" in found
    assert "iso 27001" in found
    assert "soc 2" in found
    assert "aes-256" in found
    assert "90 days" in found
    assert "99.9%" in found
    assert "v4.2" in found


def test_extract_specifics_ignores_bare_prose_and_generic_cadence():
    # "annually" / "regularly" are words, not checkable specifics — no false flags.
    assert grounding.extract_specifics("Backups run regularly and are tested annually.") == []


# ------------------------------------------------------------------- check_answer

_CONTEXT = (
    "[1] Source: InfoSec-Std v4.2\n"
    "Customer data at rest is encrypted with AES-256. Encryption keys are rotated "
    "every 90 days. The environment maintains 99.9% availability."
)


def test_grounded_answer_passes():
    answer = "Yes. Data at rest is encrypted with AES-256 and keys are rotated every 90 days."
    report = grounding.check_answer(answer, context=_CONTEXT)
    assert report.ok
    assert report.unverified == []
    assert set(map(str.lower, report.checked)) == {"aes-256", "90 days"}


def test_fabricated_specific_is_flagged():
    # The model invented a certification date and a standard absent from the evidence.
    answer = "Yes, we have been ISO 27001 certified since 2019-03-01, using AES-256 encryption."
    report = grounding.check_answer(answer, context=_CONTEXT)
    assert not report.ok
    unverified = {s.lower() for s in report.unverified}
    assert "2019-03-01" in unverified
    assert "iso 27001" in unverified
    # ...but the genuinely-grounded value is NOT flagged.
    assert "aes-256" not in unverified


def test_whitespace_insensitive_grounding():
    # Evidence line-wraps the value; the answer states it inline — still grounded.
    report = grounding.check_answer(
        "Availability is 99.9%.",
        context="Uptime target:\n99.9%\nmeasured monthly.",
    )
    assert report.ok


def test_value_echoed_from_the_question_is_grounded():
    # A specific the customer put in the question isn't a fabrication.
    report = grounding.check_answer(
        "Yes, we retain logs for 90 days.",
        context="Log retention is enforced by policy.",
        question="Do you retain audit logs for 90 days?",
    )
    assert report.ok


def test_value_from_a_past_approved_answer_is_grounded():
    report = grounding.check_answer(
        "Yes, encryption uses AES-256.",
        context="Encryption is enabled for all data at rest.",
        past_answers=[{"question": "Encryption standard?", "final_answer": "We use AES-256."}],
    )
    assert report.ok


# -------------------------------------------------------------- find_verbatim_span

def test_find_verbatim_span_exact_and_whitespace_and_reject():
    src = "Keys are rotated  every   90 days per policy."
    assert grounding.find_verbatim_span("rotated  every   90 days", src) == "rotated  every   90 days"
    # whitespace-insensitive match recovers the real span from the source
    assert grounding.find_verbatim_span("rotated every 90 days", src) == "rotated  every   90 days"
    # a span that isn't in the source is treated as fabricated
    assert grounding.find_verbatim_span("rotated every 30 days", src) is None
    assert grounding.find_verbatim_span("", src) is None


# ----------------------------------------------------------- annotate_justification

def test_annotate_prepends_marker_and_preserves_original():
    report = grounding.check_answer(
        "ISO 27001 certified since 2019-03-01.", context="No relevant evidence here."
    )
    out = grounding.annotate_justification("Grounded in [Source: InfoSec-Std].", report)
    assert out.startswith(grounding.UNVERIFIED_MARKER)
    assert "2019-03-01" in out
    assert "Grounded in [Source: InfoSec-Std]." in out  # original preserved beneath


def test_annotate_is_a_noop_when_grounded():
    clean = grounding.check_answer("Data is encrypted with AES-256.", context=_CONTEXT)
    assert grounding.annotate_justification("original rationale", clean) == "original rationale"


# ------------------------------------------- read-back (drives the UI badge)

def test_parse_and_strip_roundtrip_recovers_specifics_and_body():
    report = grounding.check_answer(
        "ISO 27001 certified since 2019-03-01.", context="No relevant evidence here."
    )
    stamped = grounding.annotate_justification("Grounded in [Source: InfoSec-Std].", report)

    assert grounding.has_unverified(stamped)
    # the exact values annotate wrote come back out, in order
    assert grounding.parse_marker(stamped) == report.unverified
    assert set(map(str.lower, grounding.parse_marker(stamped))) == {"iso 27001", "2019-03-01"}
    # stripping the marker paragraph leaves the original justification body
    assert grounding.strip_marker(stamped) == "Grounded in [Source: InfoSec-Std]."


def test_read_back_on_clean_text_is_inert():
    assert grounding.has_unverified("plain justification") is False
    assert grounding.parse_marker("plain justification") == []
    assert grounding.strip_marker("plain justification") == "plain justification"


def test_strip_marker_when_body_was_empty():
    report = grounding.check_answer("Certified 2019-03-01.", context="nothing relevant")
    stamped = grounding.annotate_justification("", report)  # no original body
    assert grounding.has_unverified(stamped)
    assert grounding.strip_marker(stamped) == ""
