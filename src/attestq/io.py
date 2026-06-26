"""Load and save questionnaires as JSON or YAML.

Keeps the question set as data, not code — so analysts can version a SIG/CAIQ/
due-diligence template in a file and hand it to the Engine. JSON needs nothing;
YAML uses the ``attestq[yaml]`` extra.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Union

from .adapters._util import require
from .models import Question, Questionnaire

Source = Union[str, Mapping]


def questionnaire_from_dict(data: Mapping) -> Questionnaire:
    """Build a Questionnaire from a plain mapping (parsed JSON/YAML)."""
    questions: List[Question] = []
    for q in data.get("questions", []):
        if "id" not in q or "prompt" not in q:
            raise ValueError("each question requires 'id' and 'prompt'")
        questions.append(
            Question(
                id=str(q["id"]),
                prompt=str(q["prompt"]),
                guidance=q.get("guidance"),
                choices=list(q["choices"]) if q.get("choices") else None,
                domain=q.get("domain"),
                weight=float(q.get("weight", 1.0)),
            )
        )
    return Questionnaire(
        id=str(data.get("id", "questionnaire")),
        title=str(data.get("title", "")),
        questions=questions,
    )


def questionnaire_to_dict(questionnaire: Questionnaire) -> Dict:
    """Serialize a Questionnaire back to a plain dict (omitting defaults)."""
    questions: List[Dict] = []
    for q in questionnaire.questions:
        item: Dict = {"id": q.id, "prompt": q.prompt}
        if q.guidance:
            item["guidance"] = q.guidance
        if q.choices:
            item["choices"] = list(q.choices)
        if q.domain:
            item["domain"] = q.domain
        if q.weight != 1.0:
            item["weight"] = q.weight
        questions.append(item)
    return {"id": questionnaire.id, "title": questionnaire.title, "questions": questions}


def load_questionnaire(source: Source) -> Questionnaire:
    """Load a Questionnaire from a mapping or a .json/.yaml/.yml file path."""
    if isinstance(source, Mapping):
        return questionnaire_from_dict(source)

    ext = os.path.splitext(source)[1].lower()
    with open(source, "r", encoding="utf-8") as fh:
        text = fh.read()
    if ext in (".yaml", ".yml"):
        yaml = require("yaml", "yaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("questionnaire file must contain a mapping at the top level")
    return questionnaire_from_dict(data)


def save_questionnaire(questionnaire: Questionnaire, path: str) -> str:
    """Write a Questionnaire to a .json/.yaml/.yml file; returns the path."""
    data = questionnaire_to_dict(questionnaire)
    ext = os.path.splitext(path)[1].lower()
    with open(path, "w", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            yaml = require("yaml", "yaml")
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, fh, indent=2)
    return path
