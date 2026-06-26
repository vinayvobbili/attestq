"""Tests for HashEmbedder, the bundled demo, and the CLI."""

from __future__ import annotations

import json
import math

from attestq import HashEmbedder
from attestq.cli import _collect_evidence, _format_from_path, main
from attestq.demo import DEMO_DOCUMENTS, demo_questionnaire


# --- HashEmbedder -------------------------------------------------------------


def test_hash_embedder_is_deterministic_and_unit_norm():
    emb = HashEmbedder(dim=128)
    a1 = emb(["encryption at rest aes-256"])[0]
    a2 = emb(["encryption at rest aes-256"])[0]
    assert a1 == a2  # stable across calls (blake2b, not builtin hash)
    assert len(a1) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)


def test_hash_embedder_overlap_scores_higher_than_disjoint():
    from attestq.store import cosine_similarity

    emb = HashEmbedder(dim=512)
    q = emb(["data encrypted at rest"])[0]
    related = emb(["all data at rest is encrypted with aes-256"])[0]
    unrelated = emb(["quarterly office snack budget planning"])[0]
    assert cosine_similarity(q, related) > cosine_similarity(q, unrelated)


# --- demo data ----------------------------------------------------------------


def test_demo_questionnaire_shape():
    qn = demo_questionnaire()
    assert len(qn) == 11
    assert all(q.choices for q in qn.questions)
    assert {q.domain for q in qn.questions}  # domains present
    assert len(DEMO_DOCUMENTS) == 6


# --- CLI ----------------------------------------------------------------------


def _stub_providers(_args):
    chat = lambda p: (  # noqa: E731 - tiny test stub
        "DETERMINATION: Met\nEVIDENCE SUMMARY: ok\nCITATIONS: 1\nNOTES: none"
    )
    return chat, HashEmbedder()


def test_cli_version(capsys):
    rc = main(["version"])
    assert rc == 0
    assert "attestq" in capsys.readouterr().out


def test_cli_no_command_prints_help():
    assert main([]) == 1


def test_cli_demo_runs(monkeypatch, capsys):
    monkeypatch.setattr("attestq.cli._build_providers", _stub_providers)
    rc = main(["demo"])
    assert rc == 0
    out = capsys.readouterr().out
    # Markdown report on stdout mentions every control id.
    for qid in ("GRC-1", "IAM-1", "DATA-1", "TPC-1"):
        assert qid in out


def test_cli_run_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr("attestq.cli._build_providers", _stub_providers)

    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({
        "id": "x", "title": "Test",
        "questions": [{"id": "ENC-1", "prompt": "Encrypted at rest?", "choices": ["Met", "Not Met"]}],
    }), encoding="utf-8")

    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "dp.txt").write_text("All data at rest is encrypted with AES-256.", encoding="utf-8")

    out = tmp_path / "report.md"
    rc = main(["run", "-q", str(qfile), "-e", str(ev), "-o", str(out), "-n", "x"])
    assert rc == 0
    assert out.exists()
    assert "ENC-1" in out.read_text(encoding="utf-8")


def test_cli_run_json_format(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("attestq.cli._build_providers", _stub_providers)
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({
        "id": "x", "title": "T",
        "questions": [{"id": "A-1", "prompt": "Encrypted?", "choices": ["Met", "Not Met"]}],
    }), encoding="utf-8")
    ev = tmp_path / "e.txt"
    ev.write_text("data is encrypted at rest with aes-256", encoding="utf-8")

    rc = main(["run", "-q", str(qfile), "-e", str(ev), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total"] == 1


# --- helpers ------------------------------------------------------------------


def test_collect_evidence_walks_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("beta", encoding="utf-8")
    docs = _collect_evidence([str(tmp_path)])
    assert {d["filename"] for d in docs} == {"a.txt", "b.md"}


def test_format_from_path():
    assert _format_from_path("r.json") == "json"
    assert _format_from_path("r.docx") == "docx"
    assert _format_from_path("r.md") == "md"
    assert _format_from_path(None) == "md"
