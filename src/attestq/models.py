"""Core data types for attestq.

These are plain dataclasses with no third-party dependencies so the kernel stays
import-light. Adapters (Chroma, Ollama, OpenAI, loaders) live behind optional
extras and never leak into this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Question:
    """A single item to be answered from evidence.

    Args:
        id: Stable identifier (e.g. "ENC-1"); used to key answers and citations.
        prompt: The question or control statement to evaluate.
        guidance: Optional reviewer guidance / what "good" evidence looks like.
        choices: Optional closed set of allowed determinations (e.g.
            ["Met", "Not Met", "Not Applicable"]). When set, the model is
            constrained to one of these and the confidence gate's fallback uses
            the *last* choice as the "insufficient evidence" determination.
        domain: Optional grouping label (e.g. "Access Control").
        weight: Optional relative importance, used by downstream risk rollups.
    """

    id: str
    prompt: str
    guidance: Optional[str] = None
    choices: Optional[Sequence[str]] = None
    domain: Optional[str] = None
    weight: float = 1.0


@dataclass
class Questionnaire:
    """An ordered collection of questions."""

    id: str
    title: str
    questions: Sequence[Question] = field(default_factory=list)

    def __iter__(self):
        return iter(self.questions)

    def __len__(self):
        return len(self.questions)


@dataclass(frozen=True)
class Hit:
    """A retrieved evidence chunk with its similarity score."""

    id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Best-effort human label for where this chunk came from."""
        return str(self.metadata.get("source") or self.metadata.get("filename") or self.id)


@dataclass(frozen=True)
class Citation:
    """A pointer back to the evidence that supports an answer."""

    source: str
    snippet: str
    score: float
    chunk_id: Optional[str] = None


@dataclass
class Answer:
    """The result of evaluating one question against an evidence corpus.

    Attributes:
        question_id: The Question.id this answers.
        determination: The model's verdict (constrained to Question.choices when
            those are set).
        summary: Short evidence-grounded rationale.
        citations: Supporting evidence chunks.
        confidence: Retrieval confidence in [0, 1] (the best evidence score after
            reranking). This drives the insufficient-evidence gate; it is
            deliberately objective and independent of the model's self-report.
        insufficient_evidence: True when the confidence gate fired and no LLM call
            was made — i.e. the corpus did not contain evidence relevant enough to
            answer. Absence of evidence is a valid, first-class result.
        raw: The unparsed model response (empty when the gate fired).
    """

    question_id: str
    determination: str
    summary: str
    citations: Sequence[Citation] = field(default_factory=list)
    confidence: float = 0.0
    insufficient_evidence: bool = False
    raw: str = ""
