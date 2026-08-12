"""Reviewer-correction signal: what the model drafted vs what a human shipped.

Any workflow that puts a draft in front of a reviewer who edits and approves it
is generating the most honest quality label it will ever have. It is not a
synthetic benchmark — it is a domain expert saying the draft was wrong, and a
daily user generates dozens of them. Most systems throw it away: the final
answer overwrites the draft in place, and the label is destroyed on contact.

This module is the measurement half. Given (draft, final) pairs plus the
citations the draft was built from, it answers three questions:

  1. How often is a draft shipped as written?  -> acceptance rate
  2. Does your confidence score predict that?  -> calibration by band
  3. Which source documents back drafts that   -> per-source trust
     survive review, and which back rewrites?

(2) is usually the one that matters most. Retrieval gates and confidence
thresholds get picked by hand and then never checked against an outcome. If
acceptance is flat across confidence bands, the score is decorative and every
threshold built on it is arbitrary; if it inverts, the score is actively
misleading. `is_calibrated` answers that in one call, and returns None rather
than a verdict when there isn't enough reviewed data to have one.

(3) is the seed of source-authority weighting — the item most RAG stacks skip.
You cannot assign trust scores to documents by hand, but reviewers assign them
implicitly every time they accept or rewrite an answer drawn from one.

Pure stdlib, no DB and no network: callers own I/O and hand records in, the same
split `grounding.py` and `quality.py` use, so this stays unit-testable in
isolation and works on drafts attestq did not produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Edit classes, coarsest signal first. The boundaries are deliberately wide —
# this measures whether a human had to intervene, not prose style.
ACCEPTED = "accepted"        # shipped as drafted (whitespace/punctuation only)
LIGHT_EDIT = "light_edit"    # reviewer tightened it; the substance held
HEAVY_EDIT = "heavy_edit"    # substantially reworked, some of the draft survived
REPLACED = "replaced"        # the draft was thrown away
UNREVIEWED = "unreviewed"    # no final answer yet — carries no signal

EDIT_CLASSES = (ACCEPTED, LIGHT_EDIT, HEAVY_EDIT, REPLACED)

# Similarity floors for each class, checked in descending order.
_ACCEPTED_FLOOR = 0.98
_LIGHT_FLOOR = 0.75
_HEAVY_FLOOR = 0.35

# Default confidence bands. The edges are a starting point that brackets the
# thresholds a retrieval stack typically has (a gate, and one or two "high
# confidence" cutoffs), so the report speaks directly to whether those numbers
# are earning their place. Pass your own via `bands=`.
DEFAULT_BANDS: Tuple[float, ...] = (0.0, 0.55, 0.72, 0.88, 1.01)


def _normalize(text: Optional[str]) -> str:
    """Collapse whitespace and case so formatting churn isn't read as an edit."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def similarity(draft: Optional[str], final: Optional[str]) -> float:
    """Normalized 0..1 similarity between a draft and the shipped answer."""
    d, f = _normalize(draft), _normalize(final)
    if not d and not f:
        return 1.0
    if not d or not f:
        return 0.0
    return SequenceMatcher(None, d, f).ratio()


def classify_edit(draft: Optional[str], final: Optional[str]) -> Tuple[str, float]:
    """Return (edit_class, similarity) for one drafted/shipped pair.

    A missing final answer is UNREVIEWED rather than REPLACED: an unfinished
    question is not a rejected one, and counting it as failure would make the
    acceptance rate a function of how much work is in flight.
    """
    if final is None or not _normalize(final):
        return UNREVIEWED, 0.0
    ratio = similarity(draft, final)
    if not _normalize(draft):
        return REPLACED, ratio
    if ratio >= _ACCEPTED_FLOOR:
        return ACCEPTED, ratio
    if ratio >= _LIGHT_FLOOR:
        return LIGHT_EDIT, ratio
    if ratio >= _HEAVY_FLOOR:
        return HEAVY_EDIT, ratio
    return REPLACED, ratio


@dataclass
class Citation:
    """One retrieved chunk that was shown to the model for an item.

    `text` is whatever the caller persisted. It is often the chunk truncated
    for storage rather than the full text the embedder saw — enough to train a
    cross-encoder on, which is the main reason to keep it.
    """

    source: str
    text: Optional[str] = None
    score: Optional[float] = None


@dataclass
class DraftOutcome:
    """One drafted item and what the reviewer did with it.

    `citations` are the retrieved chunks the draft was built from, so a rewrite
    can be attributed back to the evidence that produced it — and so the same
    record can be turned into reranker training pairs.
    """

    item_id: str
    app: str                      # which workflow produced it; free-form label
    prompt: str                   # question text / control text
    draft: Optional[str]
    final: Optional[str]
    confidence: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    top_citation_score: Optional[float] = None
    routed_to_sme: bool = False   # the confidence gate withheld a draft
    citations: List[Citation] = field(default_factory=list)
    occurred_at: Optional[str] = None   # ISO date, for temporal train/test splits

    def __post_init__(self):
        self.edit_class, self.similarity = classify_edit(self.draft, self.final)
        # Callers may pass either form; keep them consistent so the scorecard
        # and the pair exporter agree on what backed an answer.
        if self.citations and not self.sources:
            self.sources = [c.source for c in self.citations]
        if self.citations and self.top_citation_score is None:
            scores = [c.score for c in self.citations if c.score is not None]
            self.top_citation_score = max(scores) if scores else None

    @property
    def reviewed(self) -> bool:
        return self.edit_class != UNREVIEWED

    @property
    def survived(self) -> bool:
        """Shipped with at most cosmetic change — the draft did its job."""
        return self.edit_class == ACCEPTED


def outcome_from_answer(
    answer,
    question=None,
    final: Optional[str] = None,
    app: str = "attestq",
    occurred_at: Optional[str] = None,
) -> DraftOutcome:
    """Build a `DraftOutcome` from an attestq `Answer` plus what the human shipped.

    The bridge between drafting and this module: you evaluated a question, a
    reviewer edited the summary and approved it, and `final` is what they
    shipped. Pass `final=None` for an item still awaiting review — it will
    classify as UNREVIEWED and carry no signal, rather than counting as a
    rejection.

    Deliberately duck-typed rather than importing `Answer` and `Question`: it
    keeps this module free of intra-package imports (see the docstring), and it
    means records from a system that never touched attestq can be folded into
    the same scorecard as long as they expose the same handful of attributes.

    A gated answer (`insufficient_evidence`) is recorded as routed to an SME,
    which is what the gate firing actually means — a question a human now has
    to answer from scratch.
    """
    citations = [
        Citation(
            source=getattr(c, "source", "") or "",
            text=getattr(c, "snippet", None),
            score=getattr(c, "score", None),
        )
        for c in getattr(answer, "citations", []) or []
    ]
    prompt = ""
    if question is not None:
        prompt = getattr(question, "prompt", "") or ""
    item_id = getattr(answer, "question_id", "") or ""
    return DraftOutcome(
        item_id=item_id,
        app=app,
        prompt=prompt,
        draft=getattr(answer, "summary", None),
        final=final,
        confidence=getattr(answer, "confidence", None),
        citations=citations,
        routed_to_sme=bool(getattr(answer, "insufficient_evidence", False)),
        occurred_at=occurred_at,
    )


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class BandStats:
    label: str
    low: float
    high: float
    total: int = 0
    accepted: int = 0
    light: int = 0
    heavy: int = 0
    replaced: int = 0

    @property
    def acceptance_rate(self) -> Optional[float]:
        return _rate(self.accepted, self.total)

    @property
    def survival_rate(self) -> Optional[float]:
        """Accepted or lightly edited — the draft was a useful starting point."""
        return _rate(self.accepted + self.light, self.total)

    def as_dict(self) -> dict:
        return {
            "band": self.label,
            "total": self.total,
            "accepted": self.accepted,
            "light_edit": self.light,
            "heavy_edit": self.heavy,
            "replaced": self.replaced,
            "acceptance_rate": self.acceptance_rate,
            "survival_rate": self.survival_rate,
        }


def calibration_by_confidence(
    outcomes: Iterable[DraftOutcome],
    bands: Sequence[float] = DEFAULT_BANDS,
) -> List[BandStats]:
    """Bucket reviewed outcomes by draft confidence and score each bucket.

    This is the test of whether the confidence number means anything. A healthy
    result is monotonic — acceptance climbing with confidence. A flat profile
    says the score carries no information and the thresholds built on it are
    arbitrary; an inverted one says it is actively misleading.

    Outcomes with no confidence are skipped rather than bucketed at zero, which
    would manufacture a fake low-confidence failure population.
    """
    edges = list(bands)
    stats = [
        BandStats(label=f"{edges[i]:.2f}–{edges[i + 1]:.2f}", low=edges[i], high=edges[i + 1])
        for i in range(len(edges) - 1)
    ]

    for o in outcomes:
        if not o.reviewed or o.confidence is None:
            continue
        for s in stats:
            if s.low <= o.confidence < s.high:
                s.total += 1
                if o.edit_class == ACCEPTED:
                    s.accepted += 1
                elif o.edit_class == LIGHT_EDIT:
                    s.light += 1
                elif o.edit_class == HEAVY_EDIT:
                    s.heavy += 1
                else:
                    s.replaced += 1
                break
    return stats


def is_calibrated(stats: Sequence[BandStats], min_per_band: int = 10) -> Optional[bool]:
    """Is acceptance non-decreasing across populated confidence bands?

    Returns None when there isn't enough reviewed data to judge — an explicit
    "don't know" beats reporting a confident verdict off a handful of samples.

    `min_per_band` defaults to 10 deliberately: at low review volumes a single
    edit swings a band by 15-20 points, and calling a score "miscalibrated" off
    that noise would be the same overconfidence this exists to detect.
    """
    rates = [s.acceptance_rate for s in stats if s.total >= min_per_band]
    if len(rates) < 2:
        return None
    return all(b >= a - 1e-9 for a, b in zip(rates, rates[1:]))


def gate_check(
    outcomes: Iterable[DraftOutcome],
    gate: float,
) -> dict:
    """Acceptance below vs at-or-above a confidence gate.

    A cruder cut than full calibration, and much more robust at low volume: it
    pools every band on each side of the threshold, so it can say something
    useful about whether the gate is set right long before the per-band view
    has the samples to be trusted.
    """
    below = [o for o in outcomes if o.reviewed and o.confidence is not None and o.confidence < gate]
    at_or_above = [o for o in outcomes
                   if o.reviewed and o.confidence is not None and o.confidence >= gate]
    return {
        "gate": gate,
        "below_n": len(below),
        "below_acceptance": _rate(sum(1 for o in below if o.survived), len(below)),
        "above_n": len(at_or_above),
        "above_acceptance": _rate(sum(1 for o in at_or_above if o.survived), len(at_or_above)),
    }


@dataclass
class SourceStats:
    source: str
    total: int = 0
    accepted: int = 0
    rewritten: int = 0   # heavy edit or replaced

    @property
    def trust(self) -> Optional[float]:
        return _rate(self.accepted, self.total)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "total": self.total,
            "accepted": self.accepted,
            "rewritten": self.rewritten,
            "trust": self.trust,
        }


def source_trust(
    outcomes: Iterable[DraftOutcome],
    min_uses: int = 3,
) -> List[SourceStats]:
    """Per-source acceptance, i.e. which documents produce answers that survive.

    A document that repeatedly backs rewritten answers is either stale,
    ambiguous, or wrong for the questions it keeps matching — the signal we'd
    need to weight it down at retrieval time. Sources under `min_uses` are
    dropped: one rewrite is an anecdote, not a verdict on a document.
    """
    by_source: Dict[str, SourceStats] = {}
    for o in outcomes:
        if not o.reviewed:
            continue
        for src in set(o.sources):
            st = by_source.setdefault(src, SourceStats(source=src))
            st.total += 1
            if o.edit_class == ACCEPTED:
                st.accepted += 1
            elif o.edit_class in (HEAVY_EDIT, REPLACED):
                st.rewritten += 1

    ranked = [s for s in by_source.values() if s.total >= min_uses]
    # Least trustworthy first — the report is a worklist, not a leaderboard.
    ranked.sort(key=lambda s: (s.trust if s.trust is not None else 1.0, -s.total))
    return ranked


def build_scorecard(
    outcomes: Sequence[DraftOutcome],
    bands: Sequence[float] = DEFAULT_BANDS,
    min_source_uses: int = 3,
) -> dict:
    """Aggregate outcomes into the full report payload."""
    reviewed = [o for o in outcomes if o.reviewed]
    counts = {c: sum(1 for o in reviewed if o.edit_class == c) for c in EDIT_CLASSES}
    bands_stats = calibration_by_confidence(reviewed, bands)

    by_app: Dict[str, dict] = {}
    for app in sorted({o.app for o in outcomes}):
        app_reviewed = [o for o in reviewed if o.app == app]
        by_app[app] = {
            "total": sum(1 for o in outcomes if o.app == app),
            "reviewed": len(app_reviewed),
            "acceptance_rate": _rate(
                sum(1 for o in app_reviewed if o.survived), len(app_reviewed)
            ),
        }

    routed = sum(1 for o in outcomes if o.routed_to_sme)
    return {
        "totals": {
            "items": len(outcomes),
            "reviewed": len(reviewed),
            "unreviewed": len(outcomes) - len(reviewed),
            "routed_to_sme": routed,
            # Share of all items the gate refused to draft. This is the
            # throughput ceiling: every one is an item a human has to answer
            # from scratch, and it is the number a daily user feels.
            "sme_routing_rate": _rate(routed, len(outcomes)),
        },
        "gate_check": gate_check(reviewed, bands[1] if len(bands) > 1 else 0.55),
        "edit_classes": counts,
        "acceptance_rate": _rate(counts[ACCEPTED], len(reviewed)),
        "survival_rate": _rate(counts[ACCEPTED] + counts[LIGHT_EDIT], len(reviewed)),
        "by_app": by_app,
        "calibration": [s.as_dict() for s in bands_stats],
        "calibrated": is_calibrated(bands_stats),
        "source_trust": [s.as_dict() for s in source_trust(reviewed, min_source_uses)],
    }
