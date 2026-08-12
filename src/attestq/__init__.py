"""attestq — answer security questionnaires and compliance attestations from your evidence.

A model-, embedder-, and store-agnostic RAG kernel for turning a pile of evidence
documents into grounded, cited answers to a questionnaire — vendor security
reviews, SIG/CAIQ responses, audit-evidence requests, due-diligence forms.

Quick start::

    from attestq import Engine, Question

    engine = Engine(chat=my_llm, embed=my_embedder)   # inject any provider
    engine.ingest(["...evidence text..."], namespace="vendor-x")
    answer = engine.evaluate(
        Question(id="ENC-1", prompt="Is data encrypted at rest?",
                 choices=["Met", "Not Met", "Not Applicable"]),
        namespace="vendor-x",
    )
    print(answer.determination, answer.confidence, answer.citations)

Drafting is only half the job; the other half is not trusting the draft. Set
``Engine(verify=True)`` and every answer carries two reports — whether the
specifics it asserts (dates, versions, standards) actually occur in the
evidence, and whether its prose is carried by that evidence or merely restates
the question::

    engine = Engine(chat=my_llm, embed=my_embedder, verify=True)
    answer = engine.evaluate(question, namespace="vendor-x")
    if answer.needs_review:
        print(answer.grounding.unverified, answer.quality.detail())

Both checks FLAG and never rewrite, and both are deterministic — no second LLM
grading the first. See `attestq.grounding` and `attestq.quality`.

The core is dependency-free. Provider adapters (Chroma, Ollama, OpenAI,
document loaders, rerankers) ship behind optional extras.
"""

from . import feedback, grounding, quality
from .chunking import split_text
from .claim_classifier import LLMClaimClassifier, make_claim_classifier
from .embedders import HashEmbedder
from .engine import Engine
from .export import answers_to_dict, summarize, to_docx, to_json, to_markdown
from .feedback import (
    BandStats,
    DraftOutcome,
    SourceStats,
    build_scorecard,
    calibration_by_confidence,
    classify_edit,
    gate_check,
    is_calibrated,
    outcome_from_answer,
    source_trust,
)
from .grounding import GroundingReport, check_answer, extract_specifics, find_verbatim_span
from .io import (
    load_questionnaire,
    questionnaire_from_dict,
    questionnaire_to_dict,
    save_questionnaire,
)
from .models import Answer, Citation, Hit, Question, Questionnaire
from .prompts import build_eval_prompt, parse_response
from .protocols import ChatFn, EmbedFn, Reranker, VectorStore
from .quality import AnswerQualityReport, ClaimClassifier, assess, evidence_support
from .store import InMemoryVectorStore, cosine_similarity

__version__ = "0.4.0"

__all__ = [
    "Engine",
    "Question",
    "Questionnaire",
    "Answer",
    "Citation",
    "Hit",
    "InMemoryVectorStore",
    "HashEmbedder",
    "VectorStore",
    "Reranker",
    "ChatFn",
    "EmbedFn",
    "build_eval_prompt",
    "parse_response",
    "split_text",
    "cosine_similarity",
    # verification layer
    "grounding",
    "quality",
    "GroundingReport",
    "AnswerQualityReport",
    "check_answer",
    "assess",
    "extract_specifics",
    "find_verbatim_span",
    "evidence_support",
    "ClaimClassifier",
    "LLMClaimClassifier",
    "make_claim_classifier",
    # feedback / calibration
    "feedback",
    "DraftOutcome",
    "BandStats",
    "SourceStats",
    "classify_edit",
    "outcome_from_answer",
    "build_scorecard",
    "calibration_by_confidence",
    "is_calibrated",
    "gate_check",
    "source_trust",
    # export
    "to_json",
    "to_markdown",
    "to_docx",
    "summarize",
    "answers_to_dict",
    # io
    "load_questionnaire",
    "save_questionnaire",
    "questionnaire_from_dict",
    "questionnaire_to_dict",
    "__version__",
]
