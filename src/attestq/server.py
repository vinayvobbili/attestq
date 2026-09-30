"""HTTP service around an Engine, for browser clients like the attestq Chrome extension.

    pip install "attestq[server]"
    attestq serve --demo                       # sample evidence, no provider needed
    attestq serve --chroma ./store --provider openai --base-url https://my-gateway/v1

The service answers questions; it does not know about web pages. Clients send
``Question``-shaped items and get back each answer with the signals a reviewer
needs — determination, summary, citations, confidence, the insufficient-evidence
gate and any verification flags. What a reviewer finally submits comes back
through ``/feedback`` and feeds the ``attestq.feedback`` scorecard.

Endpoints:
    GET  /health       liveness + version
    GET  /namespaces   corpora the store holds (the client's vendor picker)
    POST /answer       answer questions against one namespace
    POST /feedback     record drafts vs. what reviewers shipped
    GET  /scorecard    ``build_scorecard`` over everything recorded

Browsers may only call it from ``chrome-extension://`` origins by default: a
wildcard origin would let any website the reviewer visits read answers drawn
from confidential evidence. Set ``api_token`` to require a bearer token too.
"""

import hmac
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List, Optional

from . import __version__
from .adapters._util import require
from .engine import Engine
from .feedback import Citation as OutcomeCitation
from .feedback import DraftOutcome, build_scorecard
from .models import Answer, Question

require("fastapi", "server")

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

log = logging.getLogger("attestq.server")

EXTENSION_ORIGINS = r"^chrome-extension://[a-p]{32}$"
MAX_QUESTIONS = 200


# --- wire models --------------------------------------------------------------


class QuestionIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=4000)
    choices: Optional[List[str]] = None
    guidance: Optional[str] = Field(default=None, max_length=4000)


class AnswerRequest(BaseModel):
    namespace: str = Field(min_length=1)
    questions: List[QuestionIn] = Field(max_length=MAX_QUESTIONS)


class CitationOut(BaseModel):
    source: str
    snippet: str
    score: float


class AnswerOut(BaseModel):
    question_id: str
    determination: str = ""
    summary: str = ""
    citations: List[CitationOut] = []
    confidence: float = 0.0
    insufficient_evidence: bool = False
    needs_review: bool = False
    review_notes: List[str] = []
    error: Optional[str] = None


class AnswerResponse(BaseModel):
    namespace: str
    answers: List[AnswerOut]


class FeedbackItem(BaseModel):
    question_id: str
    prompt: str = ""
    draft: Optional[str] = None
    final: Optional[str] = None
    confidence: Optional[float] = None
    insufficient_evidence: bool = False
    sources: List[str] = []


class FeedbackRequest(BaseModel):
    namespace: str = ""
    app: str = "attestq-web"
    items: List[FeedbackItem] = Field(max_length=MAX_QUESTIONS)


# --- conversions --------------------------------------------------------------


def answer_to_out(answer: Answer) -> AnswerOut:
    notes: List[str] = []
    if answer.grounding is not None and not answer.grounding.ok:
        notes.append("Not found in evidence: " + "; ".join(answer.grounding.unverified))
    if answer.quality is not None and answer.quality.flagged:
        notes.append(answer.quality.detail())
    return AnswerOut(
        question_id=answer.question_id,
        determination=answer.determination,
        summary=answer.summary,
        citations=[CitationOut(source=c.source, snippet=c.snippet, score=c.score)
                   for c in answer.citations],
        confidence=round(answer.confidence, 3),
        insufficient_evidence=answer.insufficient_evidence,
        needs_review=answer.needs_review,
        review_notes=notes,
    )


def outcome_from_item(item: FeedbackItem, app: str, occurred_at: Optional[str]) -> DraftOutcome:
    return DraftOutcome(
        item_id=item.question_id,
        app=app,
        prompt=item.prompt,
        draft=item.draft,
        final=item.final,
        confidence=item.confidence,
        sources=list(item.sources),
        routed_to_sme=item.insufficient_evidence,
        occurred_at=occurred_at,
    )


class FeedbackLog:
    """Append-only JSONL of DraftOutcomes. One line per reviewed item."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, outcomes: List[DraftOutcome], namespace: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for o in outcomes:
            record = asdict(o)
            record["namespace"] = namespace
            lines.append(json.dumps(record, default=str))
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))

    def read(self) -> List[DraftOutcome]:
        if not self.path.is_file():
            return []
        fields = set(DraftOutcome.__dataclass_fields__)
        outcomes = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["citations"] = [OutcomeCitation(**c) for c in record.get("citations") or []]
            outcomes.append(DraftOutcome(**{k: v for k, v in record.items() if k in fields}))
        return outcomes


# --- app ----------------------------------------------------------------------


def create_app(
    engine: Engine,
    *,
    api_token: Optional[str] = None,
    feedback_path: Optional[str] = None,
    allowed_origins: str = EXTENSION_ORIGINS,
    workers: int = 8,
) -> FastAPI:
    """Build the FastAPI app around a ready Engine (evidence already ingested).

    Args:
        engine: The Engine to answer with. Its store must be populated.
        api_token: When set, every call except /health needs
            ``Authorization: Bearer <token>``.
        feedback_path: JSONL file for /feedback and /scorecard; both return 404
            when unset.
        allowed_origins: Regex of browser origins CORS admits.
        workers: Questions answered concurrently per request. Each is one LLM
            call, so this bounds load on the provider as much as latency.
    """
    app = FastAPI(title="attestq", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    feedback = FeedbackLog(Path(feedback_path)) if feedback_path else None
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="attestq")

    def authorize(request: Request) -> None:
        if not api_token:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), api_token.encode()):
            raise HTTPException(401, "Missing or wrong API token.")

    def require_feedback() -> FeedbackLog:
        if feedback is None:
            raise HTTPException(404, "Feedback is not enabled on this server (start it with --feedback PATH).")
        return feedback

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "verify": engine.verify}

    @app.get("/namespaces", dependencies=[Depends(authorize)])
    def namespaces() -> dict:
        lister = getattr(engine.store, "namespaces", None)
        names = lister() if callable(lister) else []
        return {"namespaces": [{"name": n, "chunks": engine.store.count(n)} for n in names]}

    @app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(authorize)])
    def answer(req: AnswerRequest) -> AnswerResponse:
        if engine.store.count(req.namespace) == 0:
            raise HTTPException(404, f"No evidence is loaded for '{req.namespace}'.")

        def evaluate(q: QuestionIn) -> AnswerOut:
            question = Question(id=q.id, prompt=q.prompt, choices=q.choices or None,
                                guidance=q.guidance)
            try:
                return answer_to_out(engine.evaluate(question, namespace=req.namespace))
            except Exception as exc:  # one failed LLM call shouldn't sink the page
                log.exception("question %s failed", q.id)
                return AnswerOut(question_id=q.id, error=f"{type(exc).__name__}: {exc}")

        answers = list(pool.map(evaluate, req.questions))
        log.info("%s: answered %d questions", req.namespace, len(answers))
        return AnswerResponse(namespace=req.namespace, answers=answers)

    @app.post("/feedback", dependencies=[Depends(authorize)])
    def record_feedback(req: FeedbackRequest, fb: FeedbackLog = Depends(require_feedback)) -> dict:
        today = date.today().isoformat()
        fb.append([outcome_from_item(i, req.app, today) for i in req.items], req.namespace)
        return {"recorded": len(req.items)}

    @app.get("/scorecard", dependencies=[Depends(authorize)])
    def scorecard(fb: FeedbackLog = Depends(require_feedback)) -> dict:
        return build_scorecard(fb.read())

    return app
