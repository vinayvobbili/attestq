"""Tests for export renderers and questionnaire JSON/YAML loading."""

from __future__ import annotations

import importlib.util
import json

import pytest

from attestq import (
    Answer,
    Citation,
    Question,
    Questionnaire,
    answers_to_dict,
    load_questionnaire,
    questionnaire_to_dict,
    save_questionnaire,
    summarize,
    to_json,
    to_markdown,
)


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture
def sample():
    qn = Questionnaire(
        id="sec",
        title="Vendor Security Review",
        questions=[
            Question(id="ENC-1", prompt="Encrypted at rest?", choices=["Met", "Not Met"], domain="Data"),
            Question(id="IR-1", prompt="Incident response plan?", choices=["Met", "Not Met"], domain="Resilience"),
        ],
    )
    answers = [
        Answer(
            question_id="ENC-1",
            determination="Met",
            summary="AES-256 at rest.",
            citations=[Citation(source="DataProtection.pdf", snippet="AES-256", score=0.91, chunk_id="c0")],
            confidence=0.91,
        ),
        Answer(
            question_id="IR-1",
            determination="Not Met",
            summary="No evidence relevant enough to answer.",
            citations=[],
            confidence=0.2,
            insufficient_evidence=True,
        ),
    ]
    return qn, answers


def test_summarize_counts(sample):
    _, answers = sample
    s = summarize(answers)
    assert s["total"] == 2
    assert s["by_determination"] == {"Met": 1, "Not Met": 1}
    assert s["insufficient_evidence"] == 1
    assert 0.0 <= s["average_confidence"] <= 1.0


def test_to_json_roundtrips_and_enriches(sample):
    qn, answers = sample
    doc = json.loads(to_json(answers, questionnaire=qn))
    assert doc["summary"]["total"] == 2
    assert doc["questionnaire"]["id"] == "sec"
    enc = next(a for a in doc["answers"] if a["question_id"] == "ENC-1")
    assert enc["prompt"] == "Encrypted at rest?"
    assert enc["domain"] == "Data"
    assert enc["citations"][0]["source"] == "DataProtection.pdf"


def test_answers_to_dict_without_questionnaire(sample):
    _, answers = sample
    rows = answers_to_dict(answers)
    assert "prompt" not in rows[0]  # no enrichment when no questionnaire
    assert rows[0]["question_id"] == "ENC-1"


def test_to_markdown_groups_and_flags(sample):
    qn, answers = sample
    md = to_markdown(answers, questionnaire=qn)
    assert "# Vendor Security Review" in md
    assert "## Data" in md and "## Resilience" in md
    assert "ENC-1 — Met" in md
    assert "_(insufficient evidence)_" in md
    assert "DataProtection.pdf" in md


def test_load_questionnaire_from_json(tmp_path):
    data = {
        "id": "ddq",
        "title": "Due Diligence",
        "questions": [
            {"id": "ENC-1", "prompt": "Encrypted?", "choices": ["Met", "Not Met"], "domain": "Data"},
            {"id": "IAM-1", "prompt": "MFA?"},
        ],
    }
    p = tmp_path / "q.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    qn = load_questionnaire(str(p))
    assert qn.id == "ddq"
    assert len(qn) == 2
    assert qn.questions[0].choices == ["Met", "Not Met"]
    assert qn.questions[1].choices is None


def test_load_questionnaire_from_mapping_and_validation():
    qn = load_questionnaire({"id": "x", "title": "t", "questions": [{"id": "A", "prompt": "p?"}]})
    assert qn.questions[0].id == "A"
    with pytest.raises(ValueError):
        load_questionnaire({"questions": [{"id": "A"}]})  # missing prompt


def test_questionnaire_dict_roundtrip(sample):
    qn, _ = sample
    rebuilt = load_questionnaire(questionnaire_to_dict(qn))
    assert rebuilt.id == qn.id
    assert [q.id for q in rebuilt.questions] == [q.id for q in qn.questions]
    assert rebuilt.questions[0].domain == "Data"


@pytest.mark.skipif(not _installed("yaml"), reason="pyyaml not installed")
def test_questionnaire_yaml_roundtrip(tmp_path, sample):
    qn, _ = sample
    p = tmp_path / "q.yaml"
    save_questionnaire(qn, str(p))
    loaded = load_questionnaire(str(p))
    assert loaded.title == qn.title
    assert [q.id for q in loaded.questions] == [q.id for q in qn.questions]


@pytest.mark.skipif(_installed("yaml"), reason="pyyaml is installed")
def test_yaml_without_extra_raises(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("id: x\ntitle: t\nquestions: []\n", encoding="utf-8")
    with pytest.raises(ImportError) as exc:
        load_questionnaire(str(p))
    assert "attestq[yaml]" in str(exc.value)


@pytest.mark.skipif(not _installed("docx"), reason="python-docx not installed")
def test_to_docx_writes_file(tmp_path, sample):
    from attestq import to_docx

    qn, answers = sample
    out = tmp_path / "report.docx"
    to_docx(answers, str(out), questionnaire=qn)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(_installed("docx"), reason="python-docx is installed")
def test_to_docx_without_extra_raises(tmp_path, sample):
    from attestq import to_docx

    qn, answers = sample
    with pytest.raises(ImportError) as exc:
        to_docx(answers, str(tmp_path / "r.docx"), questionnaire=qn)
    assert "attestq[export]" in str(exc.value)
