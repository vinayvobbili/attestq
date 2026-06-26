"""The orchestration kernel: ingest evidence, evaluate questions.

Engine ties together the injected chat model, embedder, vector store, and
(optional) reranker. It owns the retrieve -> rerank -> confidence-gate -> draft
pipeline that turns an evidence corpus into grounded, cited answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Union

from .chunking import split_text
from .models import Answer, Citation, Hit, Question, Questionnaire
from .prompts import build_eval_prompt, parse_response
from .protocols import ChatFn, EmbedFn, Reranker, VectorStore
from .store import InMemoryVectorStore

# A document to ingest: raw text, or a (text, metadata) pair / mapping.
Document = Union[str, tuple, Mapping]

PromptBuilder = Callable[[Question, Sequence[Hit]], str]
ResponseParser = Callable[[str, Sequence[Hit]], "tuple[str, str, List[Citation]]"]

INSUFFICIENT_SUMMARY = (
    "No evidence relevant enough to answer this question was found in the corpus "
    "reviewed. Treated as insufficient evidence."
)


@dataclass
class Engine:
    """Evidence-grounded questionnaire answering.

    Args:
        chat: Any ``(prompt:str) -> str`` LLM callable.
        embed: Any ``(texts) -> list[vector]`` embedder callable.
        store: A VectorStore (defaults to in-memory).
        reranker: Optional cross-encoder reranker for precision.
        k: How many chunks to retrieve from the store per question.
        rerank_top_k: How many chunks to keep after reranking and feed to the LLM.
            Kept generously wide so a single focused document is not dropped on a
            small corpus — a precision lesson learned the hard way.
        min_confidence: Retrieval-score gate in [0, 1]. When the best available
            evidence scores below this, the question is answered "insufficient
            evidence" WITHOUT calling the LLM (no hallucination on thin air).
        gate_on: Which score the confidence gate uses — ``"retrieval"`` (default;
            the best raw similarity score *before* reranking) or ``"rerank"`` (the
            top score *after* reranking). Gate on ``"retrieval"`` when your
            ``min_confidence`` is calibrated to the embedder's similarity scale and
            the reranker only reorders for context selection; the two coincide when
            no reranker is set.
        insufficient_determination: Determination to use when the gate fires.
            Defaults to the question's last choice, else "Insufficient Evidence".
            Set this when a closed choice set's "negative" end is not the gate
            outcome you want (e.g. "Not Met" rather than "Not Applicable").
        insufficient_summary: Summary text used when the gate fires.
        chunk_size / chunk_overlap: Defaults for the built-in splitter.
        prompt_builder / response_parser: Override to change the LLM contract.
    """

    chat: ChatFn
    embed: EmbedFn
    store: VectorStore = None  # type: ignore[assignment]
    reranker: Optional[Reranker] = None
    k: int = 12
    rerank_top_k: int = 8
    min_confidence: float = 0.45
    gate_on: str = "retrieval"
    insufficient_determination: Optional[str] = None
    insufficient_summary: str = INSUFFICIENT_SUMMARY
    chunk_size: int = 1500
    chunk_overlap: int = 300
    prompt_builder: PromptBuilder = build_eval_prompt
    response_parser: ResponseParser = parse_response

    def __post_init__(self):
        if self.store is None:
            self.store = InMemoryVectorStore()
        if self.gate_on not in ("retrieval", "rerank"):
            raise ValueError("gate_on must be 'retrieval' or 'rerank'")

    # -- ingestion -------------------------------------------------------------

    def ingest(
        self,
        documents: Iterable[Document],
        namespace: str = "default",
        chunk: bool = True,
    ) -> int:
        """Embed and store evidence documents under `namespace`.

        Each document may be a plain string, a ``(text, metadata)`` tuple, or a
        mapping with a ``text``/``content`` key plus arbitrary metadata. Returns
        the number of chunks stored.
        """
        texts: List[str] = []
        metas: List[dict] = []
        for doc in documents:
            text, meta = _normalize_document(doc)
            if not text.strip():
                continue
            parts = (
                split_text(text, self.chunk_size, self.chunk_overlap) if chunk else [text]
            )
            for i, part in enumerate(parts):
                chunk_meta = dict(meta)
                chunk_meta.setdefault("source", meta.get("source") or meta.get("filename", "evidence"))
                chunk_meta["chunk_index"] = i
                texts.append(part)
                metas.append(chunk_meta)

        if not texts:
            return 0

        embeddings = self.embed(texts)
        base = self.store.count(namespace)
        ids = [f"{namespace}-{base + i}" for i in range(len(texts))]
        self.store.add(ids, texts, embeddings, metas, namespace=namespace)
        return len(texts)

    # -- evaluation ------------------------------------------------------------

    def _retrieve_raw(self, query: str, namespace: str) -> List[Hit]:
        """First-pass retrieval, sorted by similarity (before any reranking)."""
        embedding = self.embed([query])[0]
        return self.store.query(embedding, self.k, namespace=namespace)

    def _apply_rerank(self, query: str, raw_hits: List[Hit]) -> List[Hit]:
        if self.reranker and raw_hits:
            return self.reranker.rerank(query, raw_hits, self.rerank_top_k)
        return raw_hits[: self.rerank_top_k]

    def retrieve(self, query: str, namespace: str = "default") -> List[Hit]:
        """Retrieve (and rerank) the most relevant evidence for a query."""
        return self._apply_rerank(query, self._retrieve_raw(query, namespace))

    def evaluate(self, question: Question, namespace: str = "default") -> Answer:
        """Answer a single question from the evidence in `namespace`."""
        raw_hits = self._retrieve_raw(question.prompt, namespace)
        hits = self._apply_rerank(question.prompt, raw_hits)

        retrieval_conf = raw_hits[0].score if raw_hits else 0.0
        rerank_conf = hits[0].score if hits else 0.0
        confidence = retrieval_conf if self.gate_on == "retrieval" else rerank_conf

        if confidence < self.min_confidence:
            return Answer(
                question_id=question.id,
                determination=self._insufficient_determination(question),
                summary=self.insufficient_summary,
                citations=[],
                confidence=confidence,
                insufficient_evidence=True,
                raw="",
            )

        prompt = self.prompt_builder(question, hits)
        raw = self.chat(prompt)
        determination, summary, citations = self.response_parser(raw, hits)
        return Answer(
            question_id=question.id,
            determination=determination,
            summary=summary,
            citations=citations,
            confidence=confidence,
            insufficient_evidence=False,
            raw=raw,
        )

    def _insufficient_determination(self, question: Question) -> str:
        """The determination used when the confidence gate fires.

        An explicit ``insufficient_determination`` wins; otherwise the question's
        last choice (by convention the "negative" end); otherwise a generic label.
        """
        if self.insufficient_determination is not None:
            return self.insufficient_determination
        if question.choices:
            return question.choices[-1]
        return "Insufficient Evidence"

    def evaluate_all(
        self,
        questionnaire: Union[Questionnaire, Sequence[Question]],
        namespace: str = "default",
        on_answer: Optional[Callable[[Answer], None]] = None,
    ) -> List[Answer]:
        """Answer every question; `on_answer` is called after each for progress."""
        questions = list(questionnaire)
        answers: List[Answer] = []
        for q in questions:
            ans = self.evaluate(q, namespace=namespace)
            answers.append(ans)
            if on_answer:
                on_answer(ans)
        return answers


def _normalize_document(doc: Document) -> "tuple[str, dict]":
    if isinstance(doc, str):
        return doc, {}
    if isinstance(doc, tuple):
        text = doc[0]
        meta = doc[1] if len(doc) > 1 and isinstance(doc[1], Mapping) else {}
        return text, dict(meta)
    if isinstance(doc, Mapping):
        meta = dict(doc)
        text = meta.pop("text", None) or meta.pop("content", "")
        return text, meta
    raise TypeError(f"Unsupported document type: {type(doc)!r}")
