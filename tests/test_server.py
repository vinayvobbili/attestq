"""Tests for the HTTP service (`attestq serve`) the Chrome extension talks to."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from attestq import Engine, HashEmbedder, InMemoryVectorStore  # noqa: E402
from attestq.cli import offline_chat  # noqa: E402
from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE  # noqa: E402
from attestq.server import create_app  # noqa: E402

EXTENSION = "chrome-extension://" + "a" * 32
CHOICES = ["Met", "Not Met", "Not Applicable"]


def _engine(chat=offline_chat, **kw) -> Engine:
    engine = Engine(chat=chat, embed=HashEmbedder(), **kw)
    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    return engine


def _client(engine=None, **kw) -> TestClient:
    return TestClient(create_app(engine or _engine(), **kw))


def _ask(client, *questions, namespace=DEMO_NAMESPACE, **kw):
    return client.post("/answer", json={"namespace": namespace, "questions": list(questions)}, **kw)


MFA = {"id": "q1", "prompt": "Is multi-factor authentication enforced for remote and privileged access?",
       "choices": CHOICES}
UNRELATED = {"id": "q2", "prompt": "What colour is the office carpet in the lobby?"}


def test_health_reports_version_without_auth():
    res = _client(api_token="s3cret").get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_namespaces_lists_ingested_corpora_with_chunk_counts():
    body = _client().get("/namespaces").json()
    assert [n["name"] for n in body["namespaces"]] == [DEMO_NAMESPACE]
    assert body["namespaces"][0]["chunks"] > 0


def test_answer_constrains_determination_and_cites_evidence():
    ans = _ask(_client(), MFA).json()["answers"][0]
    assert ans["question_id"] == "q1"
    assert ans["determination"] in CHOICES
    assert ans["citations"] and ans["citations"][0]["source"]
    assert ans["insufficient_evidence"] is False
    assert ans["error"] is None


def test_answer_gates_questions_the_evidence_cannot_support():
    ans = _ask(_client(), UNRELATED).json()["answers"][0]
    assert ans["insufficient_evidence"] is True
    assert ans["citations"] == []


def test_answer_preserves_question_order():
    ids = [a["question_id"] for a in _ask(_client(), UNRELATED, MFA).json()["answers"]]
    assert ids == ["q2", "q1"]


def test_unknown_namespace_is_a_clear_404():
    res = _ask(_client(), MFA, namespace="nobody")
    assert res.status_code == 404
    assert "nobody" in res.json()["detail"]


def test_one_failing_question_does_not_sink_the_rest():
    def flaky(prompt: str) -> str:
        if "multi-factor" in prompt:
            raise RuntimeError("gateway timeout")
        return offline_chat(prompt)

    enc = {"id": "q3", "prompt": "Is customer data encrypted in transit and at rest?", "choices": CHOICES}
    answers = {a["question_id"]: a for a in _ask(_client(_engine(chat=flaky)), MFA, enc).json()["answers"]}
    assert "gateway timeout" in answers["q1"]["error"]
    assert answers["q3"]["error"] is None and answers["q3"]["determination"] in CHOICES


def test_verify_surfaces_review_notes():
    def overclaims(prompt: str) -> str:
        return ("DETERMINATION: Met\nEVIDENCE SUMMARY: Certified to ISO 27001 since 2011-01-01.\n"
                "CITATIONS: 1\nNOTES: none")

    ans = _ask(_client(_engine(chat=overclaims, verify=True)), MFA).json()["answers"][0]
    assert ans["needs_review"] is True
    assert any("2011-01-01" in note for note in ans["review_notes"])


def test_token_is_required_when_configured():
    client = _client(api_token="s3cret")
    assert client.get("/namespaces").status_code == 401
    assert client.get("/namespaces", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/namespaces", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_cors_admits_extensions_but_not_websites():
    client = _client()
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/answer", headers={"Origin": EXTENSION, **preflight})
    assert ok.headers.get("access-control-allow-origin") == EXTENSION
    evil = client.options("/answer", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in evil.headers


def test_feedback_is_404_when_not_enabled():
    assert _client().get("/scorecard").status_code == 404


def test_feedback_round_trips_into_the_scorecard(tmp_path):
    client = _client(feedback_path=str(tmp_path / "fb.jsonl"))
    items = [
        {"question_id": "q1", "prompt": "MFA?", "draft": "Met. MFA is enforced.",
         "final": "Met. MFA is enforced.", "confidence": 0.8, "sources": ["AccessControl.pdf"]},
        {"question_id": "q2", "prompt": "Logs?", "draft": "Met. Logs kept.",
         "final": "Not Met. No retention policy was provided.", "confidence": 0.5},
    ]
    res = client.post("/feedback", json={"namespace": DEMO_NAMESPACE, "items": items})
    assert res.json() == {"recorded": 2}

    card = client.get("/scorecard").json()
    assert card["totals"]["reviewed"] == 2
    assert card["edit_classes"]["accepted"] == 1
    assert card["by_app"]["attestq-web"]["total"] == 2


def test_in_memory_store_lists_only_non_empty_namespaces():
    store = InMemoryVectorStore()
    store.add(["b-0"], ["x"], [[1.0]], [{}], namespace="beta")
    store.add(["a-0"], ["y"], [[1.0]], [{}], namespace="alpha")
    store.clear("beta")
    assert store.namespaces() == ["alpha"]


def test_offline_chat_picks_an_allowed_choice():
    raw = offline_chat("[1] (source: a.pdf)\nMFA is on.\nDETERMINATION must be exactly one of: Yes, No.")
    assert raw.startswith("DETERMINATION: Yes\n")
    assert "MFA is on." in raw
