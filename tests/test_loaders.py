"""Tests for the document loaders (the dep-free text/markdown paths)."""

from __future__ import annotations

import os

from attestq.loaders import load_document, load_documents


def test_load_text_and_markdown(tmp_path):
    txt = tmp_path / "policy.txt"
    txt.write_text("Encryption at rest uses AES-256.", encoding="utf-8")
    md = tmp_path / "notes.md"
    md.write_text("# Heading\n\nMFA is enforced.", encoding="utf-8")

    assert "AES-256" in load_document(str(txt))
    assert "MFA is enforced" in load_document(str(md))


def test_unknown_extension_falls_back_to_text(tmp_path):
    f = tmp_path / "evidence.weirdext"
    f.write_text("some plain content", encoding="utf-8")
    assert "plain content" in load_document(str(f))


def test_load_documents_builds_ingest_ready_dicts(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("alpha evidence", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("beta evidence", encoding="utf-8")
    empty = tmp_path / "c.txt"
    empty.write_text("   \n  ", encoding="utf-8")

    docs = load_documents([str(a), str(b), str(empty)])
    assert len(docs) == 2  # empty file skipped
    names = {d["filename"] for d in docs}
    assert names == {"a.txt", "b.md"}
    assert all("source" in d and "text" in d for d in docs)
    assert os.path.basename(str(a)) == docs[0]["source"]
