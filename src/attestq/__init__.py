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

The core is dependency-free. Provider adapters (Chroma, Ollama, OpenAI,
document loaders, rerankers) ship behind optional extras.
"""

from .chunking import split_text
from .engine import Engine
from .models import Answer, Citation, Hit, Question, Questionnaire
from .prompts import build_eval_prompt, parse_response
from .protocols import ChatFn, EmbedFn, Reranker, VectorStore
from .store import InMemoryVectorStore, cosine_similarity

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "Question",
    "Questionnaire",
    "Answer",
    "Citation",
    "Hit",
    "InMemoryVectorStore",
    "VectorStore",
    "Reranker",
    "ChatFn",
    "EmbedFn",
    "build_eval_prompt",
    "parse_response",
    "split_text",
    "cosine_similarity",
    "__version__",
]
