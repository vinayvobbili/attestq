"""Deterministic grounding check for drafted answers (trust-but-verify).

A structured / heavily-instructed prompt constrains the *shape* of a drafted
answer; it does not guarantee the answer's *specifics* are real. A model can
still assert a date, version, percentage, or standard that appears nowhere in the
retrieved evidence — the single highest-liability failure for a customer-facing
compliance answer ("Yes, certified to ISO 27001 since 2019-03-01" when neither
the standard nor the date is in the evidence).

This module is the verification half. After the model drafts, we deterministically
extract the concrete, checkable values it asserted and confirm each one actually
occurs in the grounding corpus (the retrieved context + any past approved answers
+ the question itself). Anything that doesn't is reported as an *unverified
specific* so the answer can be flagged for a human before release.

The stance is "the model proposes, deterministic code disposes": a pure-function
check over strings — no LLM call, no network — so the verifier itself cannot
hallucinate. It is the opposite of LLM-checks-LLM QA, which shares the same
evidence and therefore the same failure modes.

We FLAG, we do not delete. Unlike an atomic finding (which can be dropped whole),
an answer's specifics are woven into prose; excising a token would mangle the
sentence. The right disposition is to surface the unverified claim to the
reviewer.

Pure stdlib and no intra-package imports on purpose: this stays cheaply unit-
testable, importable in isolation, and usable on drafts attestq did not produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Short, stable leading substring of UNVERIFIED_MARKER. It's the token a UI
# matches on to render a badge; keep it stable across releases so stored
# annotations stay readable.
MARKER_ANCHOR = "⚠ UNVERIFIED SPECIFICS"

# Stamped ahead of a draft's internal justification when it asserts a specific
# the evidence does not support, so the reviewer can never mistake an ungrounded
# value for a grounded one.
UNVERIFIED_MARKER = (
    MARKER_ANCHOR + " — the following value(s) were not found in the "
    "cited evidence and may be fabricated; confirm with an SME before release"
)


def _normalize(s: str) -> str:
    """Whitespace-collapsed, lower-cased form for tolerant substring matching."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def find_verbatim_span(quote: str, source: str) -> Optional[str]:
    """Return the real span from ``source`` that ``quote`` matches, else None.

    Exact substring wins; otherwise a whitespace-insensitive match, returning the
    actual span recovered from ``source`` (so what a caller surfaces is always
    real text). A quote that matches neither is treated as fabricated (None).

    This is the generic grounding primitive — the reusable core of the verify
    layer, usable wherever a model claims to be quoting a source verbatim.
    """
    if not quote:
        return None
    if quote in source:
        return quote
    nq = _normalize(quote)
    if nq and nq in _normalize(source):
        # Recover the real span so whitespace differences don't change what we show.
        pattern = re.escape(quote.strip()).replace(r"\ ", r"\s+")
        m = re.search(pattern, source, flags=re.IGNORECASE)
        return m.group(0) if m else quote
    return None


# Concrete, checkable value classes — exactly the "specific date, name, version,
# or count" a drafting prompt forbids inventing. Deliberately conservative: we
# target values an answer must be able to trace to evidence, and skip bare counts
# ("2 datacenters") which are noisy and low-stakes. A false negative (missing a
# fabrication) just leaves the un-checked behaviour; a false positive nags the
# reviewer about a real value — so we bias toward high-signal patterns.
_SPECIFIC_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                                   # ISO date 2023-11-14
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                             # 11/14/2023
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",  # Nov 14, 2023
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    r"\b\d+(?:\.\d+)?\s?%",                                     # 99.9%
    r"\$\s?\d[\d,]*(?:\.\d+)?",                                 # $1,000,000
    r"\b\d+(?:\.\d+)?\s+(?:second|minute|hour|day|week|month|year|bit|character)s?\b",  # 90 days, 256 bit
    r"\b(?:AES|SHA|RSA|TLS|SSL|HMAC)[- ]?\d{1,4}(?:\.\d+)?\b",  # AES-256, TLS 1.2
    r"\b(?:ISO|SOC|PCI|NIST|SP|FIPS)[- ]?\d{1,5}(?:-\d+)?\b",   # ISO 27001, SOC 2, NIST 800-53
    r"\bv\d+(?:\.\d+)+\b",                                      # v4.2
    r"\bversion\s+\d+(?:\.\d+)*\b",                             # version 4.2
]
_SPECIFIC_RE = re.compile("|".join(_SPECIFIC_PATTERNS), flags=re.IGNORECASE)


def extract_specifics(text: str) -> List[str]:
    """Pull the concrete, checkable values a piece of prose asserts (deduped,
    in first-seen order). See ``_SPECIFIC_PATTERNS`` for what counts."""
    seen: set = set()
    out: List[str] = []
    for m in _SPECIFIC_RE.finditer(text or ""):
        tok = m.group(0).strip()
        key = _normalize(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    return out


@dataclass
class GroundingReport:
    """Result of checking a drafted answer's specifics against its evidence."""

    checked: List[str] = field(default_factory=list)     # every specific tested
    unverified: List[str] = field(default_factory=list)  # those absent from evidence

    @property
    def ok(self) -> bool:
        return not self.unverified

    def as_dict(self) -> dict:
        return {"checked": list(self.checked), "unverified": list(self.unverified),
                "ok": self.ok}


def check_answer(
    answer: str,
    *,
    context: str = "",
    past_answers: Optional[Sequence] = None,
    question: str = "",
) -> GroundingReport:
    """Verify every specific in ``answer`` against the grounding corpus.

    The corpus is the union of what legitimately grounds an answer: the retrieved
    ``context`` shown to the model, any ``past_answers`` (dicts with
    ``final_answer`` / ``question``, or plain strings) folded into the draft, and
    the ``question`` itself (a value echoed from the question is not a
    fabrication). A specific absent from all of them is reported as unverified.
    """
    corpus_parts: List[str] = [context or "", question or ""]
    for pa in past_answers or []:
        if isinstance(pa, dict):
            corpus_parts.append(str(pa.get("final_answer", "")))
            corpus_parts.append(str(pa.get("question", "")))
        else:
            corpus_parts.append(str(pa))
    grounding_norm = _normalize("\n".join(corpus_parts))

    report = GroundingReport()
    for specific in extract_specifics(answer):
        report.checked.append(specific)
        if _normalize(specific) not in grounding_norm:
            report.unverified.append(specific)
    return report


def annotate_justification(justification: str, report: GroundingReport) -> str:
    """Prepend the unverified-specifics marker to a draft's internal justification.

    A no-op when the report is clean, so a grounded draft's rationale is untouched.
    The marker leads (reviewers read top-down) and the original justification is
    preserved beneath it, never replaced.
    """
    if report.ok:
        return justification
    note = f"{UNVERIFIED_MARKER}: {'; '.join(report.unverified)}"
    return f"{note}\n\n{justification}".strip()


# --- Read-back: recover what annotate_justification() stamped ------------------
# The marker is persisted inside the justification / notes text (no schema
# change required of the caller), so a UI reads it back from there. These parse
# that stamp so a badge and a clean body can be rendered from one stored field.

def has_unverified(text: str) -> bool:
    """True if `text` carries an unverified-specifics marker."""
    return bool(text) and MARKER_ANCHOR in text


def parse_marker(text: str) -> List[str]:
    """Recover the unverified specifics stamped by ``annotate_justification``.

    Returns the list of values (empty if no marker is present) — the inverse of
    the ``f"{UNVERIFIED_MARKER}: a; b"`` format annotate writes.
    """
    if not text or UNVERIFIED_MARKER not in text:
        return []
    after = text.split(UNVERIFIED_MARKER, 1)[1]   # everything past the marker text
    line = after.split("\n\n", 1)[0]              # the marker paragraph only
    line = line.lstrip(":").strip()               # drop the "': '" separator
    return [s.strip() for s in line.split(";") if s.strip()]


def strip_marker(text: str) -> str:
    """Return `text` without a leading marker paragraph (the original body).

    Only strips when the marker leads the text (as annotate writes it), so a
    justification that merely mentions the token elsewhere is left intact.
    """
    if not text or not text.startswith(UNVERIFIED_MARKER):
        return text
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""
