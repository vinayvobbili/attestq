"""Render evaluated answers into shareable artifacts.

JSON and Markdown need nothing beyond the standard library. The Word (.docx)
renderer uses ``attestq[export]`` — handy when the deliverable is a form a human
reviewer signs off on.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .adapters._util import require
from .models import Answer, Question, Questionnaire


def summarize(answers: Sequence[Answer]) -> Dict:
    """Roll up a set of answers into headline counts."""
    by_det = Counter(a.determination for a in answers)
    insufficient = sum(1 for a in answers if a.insufficient_evidence)
    avg_conf = (sum(a.confidence for a in answers) / len(answers)) if answers else 0.0
    return {
        "total": len(answers),
        "by_determination": dict(by_det),
        "insufficient_evidence": insufficient,
        "average_confidence": round(avg_conf, 3),
    }


def answers_to_dict(
    answers: Sequence[Answer],
    questionnaire: Optional[Questionnaire] = None,
) -> List[Dict]:
    """Serialize answers to plain dicts, enriched with question text when known."""
    qmap = _question_map(questionnaire)
    out: List[Dict] = []
    for a in answers:
        d = asdict(a)
        q = qmap.get(a.question_id)
        if q is not None:
            d["prompt"] = q.prompt
            d["domain"] = q.domain
        out.append(d)
    return out


def to_json(
    answers: Sequence[Answer],
    questionnaire: Optional[Questionnaire] = None,
    meta: Optional[Mapping] = None,
    indent: int = 2,
) -> str:
    """Render a complete result document as a JSON string."""
    payload: Dict = {
        "summary": summarize(answers),
        "answers": answers_to_dict(answers, questionnaire),
    }
    if questionnaire is not None:
        payload["questionnaire"] = {"id": questionnaire.id, "title": questionnaire.title}
    if meta:
        payload["meta"] = dict(meta)
    return json.dumps(payload, indent=indent, default=str)


def to_markdown(
    answers: Sequence[Answer],
    questionnaire: Optional[Questionnaire] = None,
    title: Optional[str] = None,
) -> str:
    """Render a human-readable Markdown report, grouped by domain when available."""
    qmap = _question_map(questionnaire)
    heading = title or (questionnaire.title if questionnaire else "Assessment Results")
    summary = summarize(answers)

    lines: List[str] = [f"# {heading}", ""]
    lines.append(f"**Questions:** {summary['total']}  ")
    lines.append(f"**Average evidence confidence:** {summary['average_confidence']:.2f}  ")
    if summary["insufficient_evidence"]:
        lines.append(f"**Insufficient evidence:** {summary['insufficient_evidence']}  ")
    det_bits = ", ".join(f"{k}: {v}" for k, v in summary["by_determination"].items())
    lines.append(f"**Determinations:** {det_bits}")
    lines.append("")

    for domain, group in _group_by_domain(answers, qmap).items():
        if domain:
            lines.append(f"## {domain}")
            lines.append("")
        for a in group:
            q = qmap.get(a.question_id)
            prompt = q.prompt if q else a.question_id
            lines.append(f"### {a.question_id} — {a.determination}")
            lines.append(f"*{prompt}*")
            lines.append("")
            conf = f"{a.confidence:.2f}"
            flag = " _(insufficient evidence)_" if a.insufficient_evidence else ""
            lines.append(f"Confidence: {conf}{flag}")
            lines.append("")
            lines.append(a.summary or "_No summary._")
            lines.append("")
            if a.citations:
                lines.append("Evidence:")
                for c in a.citations:
                    snippet = c.snippet.replace("\n", " ").strip()
                    lines.append(f"- {c.source} ({c.score:.2f}): {snippet}")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_docx(
    answers: Sequence[Answer],
    path: str,
    questionnaire: Optional[Questionnaire] = None,
    title: Optional[str] = None,
) -> str:
    """Write a Word document report to `path` and return the path."""
    docx = require("docx", "export")
    qmap = _question_map(questionnaire)
    heading = title or (questionnaire.title if questionnaire else "Assessment Results")
    summary = summarize(answers)

    doc = docx.Document()
    doc.add_heading(heading, level=0)
    p = doc.add_paragraph()
    p.add_run(f"Questions: {summary['total']}    ")
    p.add_run(f"Average evidence confidence: {summary['average_confidence']:.2f}    ")
    if summary["insufficient_evidence"]:
        p.add_run(f"Insufficient evidence: {summary['insufficient_evidence']}")
    doc.add_paragraph(
        "Determinations — "
        + ", ".join(f"{k}: {v}" for k, v in summary["by_determination"].items())
    )

    for domain, group in _group_by_domain(answers, qmap).items():
        if domain:
            doc.add_heading(domain, level=1)
        for a in group:
            q = qmap.get(a.question_id)
            prompt = q.prompt if q else a.question_id
            doc.add_heading(f"{a.question_id} — {a.determination}", level=2)
            doc.add_paragraph(prompt, style="Intense Quote")
            conf = f"Confidence: {a.confidence:.2f}"
            if a.insufficient_evidence:
                conf += " (insufficient evidence)"
            doc.add_paragraph(conf)
            doc.add_paragraph(a.summary or "No summary.")
            if a.citations:
                doc.add_paragraph("Evidence:")
                for c in a.citations:
                    snippet = c.snippet.replace("\n", " ").strip()
                    doc.add_paragraph(f"{c.source} ({c.score:.2f}): {snippet}", style="List Bullet")

    doc.save(path)
    return path


def _question_map(questionnaire: Optional[Questionnaire]) -> Dict[str, Question]:
    if questionnaire is None:
        return {}
    return {q.id: q for q in questionnaire.questions}


def _group_by_domain(
    answers: Sequence[Answer],
    qmap: Mapping[str, Question],
) -> "OrderedDict[Union[str, None], List[Answer]]":
    grouped: "OrderedDict[Union[str, None], List[Answer]]" = OrderedDict()
    for a in answers:
        q = qmap.get(a.question_id)
        domain = q.domain if q else None
        grouped.setdefault(domain, []).append(a)
    return grouped
