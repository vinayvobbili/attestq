"""Default evidence-grounded prompt and response parser.

The prompt encodes the rules that make questionnaire auto-fill trustworthy:
answer ONLY from the supplied evidence, cite it, and say so plainly when the
evidence does not support a conclusion. You can replace the whole thing by
passing your own ``prompt_builder`` to the Engine.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .models import Citation, Hit, Question

SYSTEM_INSTRUCTION = (
    "You are an evidence analyst. Answer each question STRICTLY from the supplied "
    "evidence excerpts. Do not use outside knowledge or assumptions. If the "
    "evidence does not clearly support a conclusion, say so explicitly rather "
    "than guessing — absence of evidence is itself a valid finding. Always ground "
    "your answer in the specific excerpts provided and reference them."
)

_FIELDS = ("DETERMINATION", "EVIDENCE SUMMARY", "CITATIONS", "NOTES")


def build_eval_prompt(question: Question, hits: Sequence[Hit]) -> str:
    """Render the full prompt (system instruction + evidence + question)."""
    lines: List[str] = [SYSTEM_INSTRUCTION, ""]

    lines.append("=== EVIDENCE EXCERPTS ===")
    if hits:
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] (source: {h.source})")
            lines.append(h.text.strip())
            lines.append("")
    else:
        lines.append("(no evidence excerpts were retrieved)")
        lines.append("")

    lines.append("=== QUESTION ===")
    lines.append(question.prompt.strip())
    if question.guidance:
        lines.append("")
        lines.append(f"Reviewer guidance: {question.guidance.strip()}")

    if question.choices:
        allowed = ", ".join(question.choices)
        lines.append("")
        lines.append(f"DETERMINATION must be exactly one of: {allowed}.")

    lines.append("")
    lines.append("Respond using EXACTLY these labelled fields, each on its own line:")
    lines.append("DETERMINATION: <your verdict>")
    lines.append("EVIDENCE SUMMARY: <2-4 sentences grounded only in the excerpts>")
    lines.append("CITATIONS: <comma-separated excerpt numbers you relied on, e.g. 1, 3>")
    lines.append("NOTES: <gaps, caveats, or 'none'>")
    return "\n".join(lines)


def parse_response(raw: str, hits: Sequence[Hit]) -> Tuple[str, str, List[Citation]]:
    """Parse a labelled model response into (determination, summary, citations).

    Tolerant of missing fields and stray formatting. Citation numbers are mapped
    back onto the hit list so each Citation carries the real source and snippet.
    """
    fields = _extract_fields(raw)
    determination = fields.get("DETERMINATION", "").strip() or "Undetermined"
    summary = fields.get("EVIDENCE SUMMARY", "").strip()
    if not summary:
        # Fall back to the raw text so nothing is silently lost.
        summary = raw.strip()[:1000]

    citations = _resolve_citations(fields.get("CITATIONS", ""), hits)
    notes = fields.get("NOTES", "").strip()
    if notes and notes.lower() not in ("none", "n/a", "-"):
        summary = f"{summary}\n\nNotes: {notes}" if summary else notes
    return determination, summary, citations


def _extract_fields(raw: str) -> dict:
    out: dict = {}
    # Build a regex that captures the text between one field label and the next.
    label_alt = "|".join(re.escape(f) for f in _FIELDS)
    pattern = re.compile(
        rf"(?P<label>{label_alt})\s*[:\-]\s*(?P<value>.*?)(?=(?:\n\s*(?:{label_alt})\s*[:\-])|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(raw):
        out[m.group("label").upper()] = m.group("value").strip()
    return out


def _resolve_citations(value: str, hits: Sequence[Hit]) -> List[Citation]:
    citations: List[Citation] = []
    seen = set()
    for num in re.findall(r"\d+", value or ""):
        idx = int(num) - 1
        if 0 <= idx < len(hits) and idx not in seen:
            seen.add(idx)
            h = hits[idx]
            citations.append(
                Citation(
                    source=h.source,
                    snippet=h.text.strip()[:400],
                    score=h.score,
                    chunk_id=h.id,
                )
            )
    return citations
