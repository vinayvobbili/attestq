"""Referenceless answer-quality checks: is the answer grounded, or just an echo?

`grounding.py` verifies the *specifics* a draft asserts — dates, versions,
percentages, standards — against the retrieved evidence. That catches the
loudest hallucination (a fabricated certification date) but it is silent on the
quieter failure:

    Q: "Are encryption keys securely managed, restricted, and controlled
        through defined key-management processes?"
    A: "Yes. Encryption keys are securely managed, restricted, and controlled
        through defined key-management processes."

That answer contains no checkable specific, so `grounding.check_answer` returns
a clean report and the draft ships. But it asserts nothing the question didn't
already say — it is the question with "Yes." in front, and the recipient
receives an unsupported affirmation.

Two measurements catch it, per sentence of the answer:

  support  — how close the sentence sits to the retrieved evidence. The RAG
             Triad's GROUNDEDNESS, measured over what the answer *adds* rather
             than over the whole answer: score the whole thing and question-echo
             inflates it toward whatever the question already shared with the
             corpus.

  echo     — how close the sentence sits to the question itself. A sentence
             nearer the question than to anything retrieved is a restatement.
             This is the degenerate case of ANSWER RELEVANCE: we cannot
             referencelessly prove prose answers a question, but we can prove
             when it only repeats it.

The Triad's third leg, CONTEXT RELEVANCE, is already covered — it is the
retrieval/rerank score that `Engine`'s confidence gate thresholds on. So this
completes the deterministic subset of the Triad.

WHY NOT AN LLM JUDGE. Ragas/TruLens-style scoring prompts a model to rate its
own kind of output; it shares the drafting model's blind spots and can
hallucinate the verdict. Embedding similarity is a model too, but an
inference-only one: it emits vectors, not claims, so the verifier itself cannot
invent a justification. Same "the model proposes, deterministic code disposes"
line as `grounding.py`.

There is one judgment here that measurement genuinely cannot settle: whether a
question even wants documentary evidence. "What is the legal entity name?"
answered "The legal entity name is Acme Corp." is a fine answer that no evidence
backs, and it is indistinguishable by similarity from an ungrounded compliance
claim. That is a semantic call, and the place for a model that reasons — see the
caller-supplied `claim_classifier` hook below, which may only ever DOWNGRADE a
flag. `attestq.claim_classifier` ships one built on any chat model.

We FLAG, never rewrite: a flagged answer may still be the right answer, so the
disposition is to tell the reviewer what the draft did not earn.

No intra-package imports, so this stays importable in isolation and usable on
drafts attestq did not produce. The embedding function is injected rather than
imported for the same reason — and so tests can run without a live endpoint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

# Short, stable leading substring of WEAK_SUPPORT_MARKER — the token a UI matches
# on to render a badge. Keep it stable across releases so stored annotations stay
# readable.
MARKER_ANCHOR = "⚠ WEAK EVIDENCE SUPPORT"

WEAK_SUPPORT_MARKER = (
    MARKER_ANCHOR + " — this answer is not carried by the evidence it cites; "
    "confirm it against a source before release"
)

# Verdicts, best to worst.
SUPPORTED = "supported"        # substance present and backed by the evidence
THIN = "thin"                  # substance present, only partly traceable
UNSUPPORTED = "unsupported"    # says something the evidence does not
VACUOUS = "vacuous"            # every sentence restates the question
NO_EVIDENCE = "no_evidence"    # nothing was retrieved — unscoreable, not a pass
UNSCORED = "unscored"          # no answer text to judge
NOT_APPLICABLE = "not_applicable"   # a claim classifier cleared it

FLAGGED_VERDICTS = (UNSUPPORTED, VACUOUS)

# A sentence counts as supported when it clears BOTH scale-free tests:
#
#   1. it sits closer to the evidence than the QUESTION does. The question's own
#      similarity to its retrieved evidence is the natural per-question baseline
#      — it is what "topically related to this material" scores before anyone
#      answers anything. A sentence that beats it has drawn on the evidence;
#      one that does not is riding the question's topic.
#   2. it sits closer to the evidence than to the question. A restatement runs
#      echo 0.92-0.95 against support 0.53-0.59; a sentence genuinely lifted
#      from evidence inverts that.
#
# Deriving the baseline per question is what keeps a hardcoded cosine floor out
# of this file. Absolute similarity is model-specific and drifts the moment the
# embedding model changes; observed live, unrelated business English and a real
# quotation sat 0.62 vs 0.64 — far too close to separate with a constant, while
# their per-question baselines (0.735 and 0.544) separate them cleanly.
#
# Note what is deliberately NOT checked here: whether the evidence was any good
# in the first place. That is the retrieval gate's job (the score `Engine`
# thresholds on) and re-litigating it here would double-count.
MIN_SUPPORT = 0.50        # share of scoreable sentences that must be supported
WEAK_SUPPORT = 0.20       # below this share, the answer is not drawn from its evidence

# A sentence must have some substance before its similarity means anything.
# "Yes." is three characters and will sit near-equidistant from everything.
MIN_SENTENCE_CHARS = 25

# Inline citation markers a drafting prompt asks for — "[Source: P1]", "[3]".
# Parsing our own output format, not encoding English.
_CITATION_MARKER_RE = re.compile(
    r"\[\s*(?:source\s*[:\-]?[^\]]*|p?\d+(?:\s*[,;]\s*p?\d+)*)\s*\]",
    flags=re.IGNORECASE,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Structurally identical to `attestq.protocols.EmbedFn`; redeclared rather than
# imported to keep this module free of intra-package imports (see the docstring).
EmbedFn = Callable[[List[str]], List[List[float]]]
ClaimClassifier = Callable[[str, str], bool]   # (question, answer) -> needs evidence?


def sentences(text: str) -> List[str]:
    """Split prose into sentences on terminal punctuation and line breaks."""
    clean = _CITATION_MARKER_RE.sub(" ", text or "")
    return [s.strip() for s in _SENTENCE_RE.split(clean) if s.strip()]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Mirrors `attestq.store.cosine_similarity`; kept local so
    this module has no intra-package imports."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _ratio(hit: int, total: int) -> float:
    return round(hit / total, 4) if total else 0.0


@dataclass
class SentenceScore:
    """One answer sentence, and where it sits between question and evidence."""

    text: str
    support: float = 0.0    # max similarity to any evidence chunk
    echo: float = 0.0       # similarity to the question
    baseline: float = 0.0   # the question's own similarity to that evidence

    @property
    def is_echo(self) -> bool:
        """Closer to the question than to anything we retrieved."""
        return self.echo > self.support

    @property
    def is_supported(self) -> bool:
        """Drawn from the evidence rather than from the question.

        The `support > 0` guard is not redundant: with no question to compare
        against, the baseline is zero, and without it a sentence sharing nothing
        at all with the evidence would clear `support >= baseline` at 0 >= 0.
        """
        return self.support > 0.0 and self.support >= self.baseline and not self.is_echo

    def as_dict(self) -> dict:
        return {"text": self.text[:160], "support": round(self.support, 4),
                "echo": round(self.echo, 4), "baseline": round(self.baseline, 4),
                "is_echo": self.is_echo, "is_supported": self.is_supported}


@dataclass
class AnswerQualityReport:
    """How much of an answer is the question echoed back, and how much of the
    rest is actually in the evidence."""

    verdict: str = UNSCORED
    echo: float = 0.0                 # share of scoreable sentences that echo
    support: float = 0.0              # share of scoreable sentences carried by evidence
    scored_sentences: int = 0
    unsupported: List[str] = field(default_factory=list)   # the sentences that failed
    has_evidence: bool = False
    method: str = "none"              # "embedding" | "lexical" | "none"
    # What measurement concluded before a claim classifier overrode it. Empty
    # unless a downgrade happened. Without this the override is unauditable:
    # NOT_APPLICABLE alone cannot distinguish "correctly cleared an entity-name
    # lookup" from "cleared a vacuous answer to a real control question".
    measured: str = ""

    @property
    def ok(self) -> bool:
        """True when nothing needs a reviewer's attention.

        NO_EVIDENCE and UNSCORED count as ok: they mean this check had nothing
        to judge, which is the retrieval gate's problem to report, not ours.
        Silence from a check that could not run must never read as a pass, so
        callers get the verdict alongside.
        """
        return self.verdict not in FLAGGED_VERDICTS

    @property
    def flagged(self) -> bool:
        return not self.ok

    def detail(self) -> str:
        """One line naming what the draft failed to earn."""
        if self.verdict == VACUOUS:
            return ("every sentence of the answer is closer to the question than "
                    "to any retrieved evidence — it restates the question rather "
                    "than answering it from a source")
        if self.verdict == UNSUPPORTED:
            first = self.unsupported[0][:140] if self.unsupported else ""
            return (
                f"only {self.support:.0%} of the answer is carried by the cited "
                f"evidence" + (f'; unsupported: "{first}"' if first else "")
            )
        if self.verdict == THIN:
            return (f"{self.support:.0%} of the answer is traceable to the cited "
                    "evidence")
        if self.verdict == NO_EVIDENCE:
            return "no evidence was cited, so support could not be checked"
        if self.verdict == NOT_APPLICABLE:
            was = f" (measured {self.measured}, {self.support:.0%} carried)" if self.measured else ""
            return ("administrative question — no documentary evidence expected"
                    + was)
        return f"{self.support:.0%} of the answer is carried by the evidence"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "echo": self.echo,
            "support": self.support,
            "scored_sentences": self.scored_sentences,
            "unsupported": [s[:160] for s in self.unsupported],
            "has_evidence": self.has_evidence,
            "method": self.method,
            "measured": self.measured,
        }


def _evidence_units(context: str, past_answers: Optional[Sequence]) -> List[str]:
    """Everything that may legitimately ground an answer, as comparable units.

    Note what is absent: the question. `grounding.check_answer` folds it in (a
    specific echoed from the question is not fabricated), but here the question
    is the baseline we measure novelty against — counting it as evidence would
    make every restatement perfectly supported, the exact failure this exists to
    catch.
    """
    parts: List[str] = []
    for block in re.split(r"\n\s*\n", context or ""):
        if block.strip():
            parts.append(block.strip())
    for pa in past_answers or []:
        if isinstance(pa, dict):
            joined = f"{pa.get('question', '')}\n{pa.get('final_answer', '')}".strip()
            if joined:
                parts.append(joined)
        elif str(pa).strip():
            parts.append(str(pa).strip())
    return parts


# --- Lexical fallback ---------------------------------------------------------
# Used when no embedding function is supplied or reachable, and by offline batch
# tools that would rather not embed tens of thousands of rows. Crude on purpose:
# it borrows its stopword list and stemmer from libraries that already maintain
# them (if installed), rather than growing a hand-tuned table here that silently
# rots. Both imports are optional — absent, it degrades to no stopwords and no
# stemming, which still works because every comparison uses the same measure.

def _lexical_tools():
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _sw
        stops = set(_sw)
    except Exception:
        stops = set()
    try:
        from nltk.stem import PorterStemmer
        stem = PorterStemmer().stem
    except Exception:
        def stem(w):        # noqa: E306 - trivial identity fallback
            return w
    return stops, stem


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-/.]*")


def _terms(text: str, stops, stem) -> set:
    out = set()
    for m in _WORD_RE.finditer((text or "").lower()):
        tok = m.group(0).strip("-/.")
        if len(tok) < 3 or tok in stops or tok.isdigit():
            continue
        out.add(stem(tok))
    return out


def _lexical_scores(answer_sents, question, evidence) -> List[SentenceScore]:
    """Term overlap standing in for cosine, on its own self-consistent scale.

    The baseline comparison is what makes swapping the similarity measure safe:
    both sides of every test are computed the same way, so the units cancel.
    """
    stops, stem = _lexical_tools()
    q_terms = _terms(question, stops, stem)
    ev_terms = [_terms(e, stops, stem) for e in evidence]

    def overlap(a: set, b: set) -> float:
        return len(a & b) / len(a) if a else 0.0

    baseline = max((overlap(q_terms, e) for e in ev_terms), default=0.0)
    scored = []
    for sent in answer_sents:
        s_terms = _terms(sent, stops, stem)
        scored.append(SentenceScore(
            text=sent,
            support=max((overlap(s_terms, e) for e in ev_terms), default=0.0),
            echo=overlap(s_terms, q_terms),
            baseline=baseline,
        ))
    return scored


def _embedded_scores(answer_sents, question, evidence, embed_fn) -> List[SentenceScore]:
    vectors = embed_fn([question] + evidence + answer_sents)
    q_vec = vectors[0]
    ev_vecs = vectors[1:1 + len(evidence)]
    a_vecs = vectors[1 + len(evidence):]
    baseline = max((cosine(q_vec, e) for e in ev_vecs), default=0.0)
    return [
        SentenceScore(
            text=sent,
            support=max((cosine(vec, e) for e in ev_vecs), default=0.0),
            echo=cosine(vec, q_vec),
            baseline=baseline,
        )
        for sent, vec in zip(answer_sents, a_vecs)
    ]


def assess(
    answer: str,
    *,
    context: str = "",
    question: str = "",
    past_answers: Optional[Sequence] = None,
    embed_fn: Optional[EmbedFn] = None,
    claim_classifier: Optional[ClaimClassifier] = None,
    min_support: float = MIN_SUPPORT,
    weak_support: float = WEAK_SUPPORT,
) -> AnswerQualityReport:
    """Score one drafted answer against its question and its cited evidence.

    Signature deliberately mirrors ``grounding.check_answer`` so both checks wire
    into a drafting path the same way. ``embed_fn`` takes a list of strings and
    returns their vectors; without one (or if it raises) the lexical fallback
    runs and says so in ``report.method``.

    ``claim_classifier(question, answer) -> bool`` may clear a flag when the
    question wants no documentary evidence. It can only downgrade: a model that
    is wrong here costs a nag, never a missed ungrounded claim.
    """
    report = AnswerQualityReport()
    all_sents = sentences(answer)
    if not all_sents:
        return report

    # Fragments too short to carry a claim ("Yes.") sit near-equidistant from
    # everything, so scoring them would be noise either way.
    scoreable = [s for s in all_sents if len(s) >= MIN_SENTENCE_CHARS]
    evidence = _evidence_units(context, past_answers)
    report.has_evidence = bool(evidence)

    if not scoreable:
        # The whole answer is fragments. Only meaningful as a restatement if it
        # tracks the question; otherwise it is simply terse ("United States").
        report.verdict = UNSCORED
        return report
    if not evidence:
        report.verdict = NO_EVIDENCE
        return report

    scored: List[SentenceScore] = []
    if embed_fn is not None:
        try:
            scored = _embedded_scores(scoreable, question, evidence, embed_fn)
            report.method = "embedding"
        except Exception:
            scored = []
    if not scored:
        scored = _lexical_scores(scoreable, question, evidence)
        report.method = "lexical"

    report.scored_sentences = len(scored)
    supported = [s for s in scored if s.is_supported]
    report.support = _ratio(len(supported), len(scored))
    report.echo = _ratio(sum(1 for s in scored if s.is_echo), len(scored))
    report.unsupported = [s.text for s in scored if not s.is_supported]

    if all(s.is_echo for s in scored) and not supported:
        report.verdict = VACUOUS
    elif report.support >= min_support:
        report.verdict = SUPPORTED
    elif report.support >= weak_support:
        report.verdict = THIN
    else:
        report.verdict = UNSUPPORTED

    # Last word goes to the judgment measurement can't make: does this question
    # want evidence at all? Downgrade-only, so a wrong call costs nothing worse
    # than the un-classified behaviour.
    if report.flagged and claim_classifier is not None:
        try:
            if not claim_classifier(question, answer):
                report.measured = report.verdict
                report.verdict = NOT_APPLICABLE
        except Exception:
            pass
    return report


def evidence_support(
    answer: str,
    passage: str,
    question: str = "",
    embed_fn: Optional[EmbedFn] = None,
) -> float:
    """Share of an answer carried by one passage, 0.0–1.0.

    The single-passage form, for offline use: deciding whether a retrieved chunk
    actually contributed to a shipped answer (and is therefore a real positive
    training pair) rather than merely co-occurring with it.
    """
    r = assess(answer, context=passage, question=question, embed_fn=embed_fn)
    return 0.0 if r.verdict in (VACUOUS, UNSCORED, NO_EVIDENCE) else r.support


# --- Read-back: recover what annotate() stamped -------------------------------
# Mirrors grounding.py exactly. The marker is persisted inside the existing
# justification / notes text — no schema change required of the caller — and a UI
# parses it back out to render a badge over a clean body.

def annotate(justification: str, report: AnswerQualityReport) -> str:
    """Prepend the weak-support marker to a draft's internal justification.

    A no-op for an unflagged report, so a well-grounded draft's rationale is
    untouched. The marker leads and the original text is preserved beneath it.
    """
    if report.ok:
        return justification or ""
    note = f"{WEAK_SUPPORT_MARKER}: {report.detail()}"
    return f"{note}\n\n{justification}".strip()


def has_weak_support(text: str) -> bool:
    """True if `text` carries a weak-support marker."""
    return bool(text) and MARKER_ANCHOR in text


def parse_marker(text: str) -> str:
    """Recover the detail line stamped by ``annotate`` ("" if absent)."""
    if not text or WEAK_SUPPORT_MARKER not in text:
        return ""
    after = text.split(WEAK_SUPPORT_MARKER, 1)[1]
    return after.split("\n\n", 1)[0].lstrip(":").strip()


def strip_marker(text: str) -> str:
    """Return `text` without a leading marker paragraph (the original body)."""
    if not text or not text.startswith(WEAK_SUPPORT_MARKER):
        return text
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""
