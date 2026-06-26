"""Minimal web demo for attestq.

A single-page Flask app that runs the bundled Helios sample assessment and shows
the result — determinations, confidence, citations, and the insufficient-evidence
gate. Runs with zero setup in DEMO mode (built-in HashEmbedder + placeholder
chat); set a provider to get real reasoned determinations.

    pip install attestq flask
    python examples/web/app.py                 # demo mode
    OPENAI_API_KEY=sk-... python examples/web/app.py   # live mode
    PROVIDER=ollama python examples/web/app.py          # local Ollama

Then open http://localhost:5000
"""

import os

from flask import Flask, render_template, request

from attestq import Engine, HashEmbedder, to_markdown
from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE, demo_questionnaire

app = Flask(__name__)


def _placeholder_chat(prompt: str) -> str:
    return (
        "DETERMINATION: Met\n"
        "EVIDENCE SUMMARY: Placeholder verdict — connect a real LLM (OPENAI_API_KEY "
        "or PROVIDER=ollama) for an evidence-reasoned determination. Retrieval, "
        "citations, and the insufficient-evidence gate shown here are all real.\n"
        "CITATIONS: 1\n"
        "NOTES: demo mode"
    )


def build_engine():
    """Return (engine, mode). Live if a provider is configured, else demo."""
    provider = os.environ.get("PROVIDER", "openai" if os.environ.get("OPENAI_API_KEY") else None)
    if provider == "openai":
        from attestq.adapters import OpenAIChat, OpenAIEmbedder
        return Engine(chat=OpenAIChat(model="gpt-4o-mini"), embed=OpenAIEmbedder()), "live"
    if provider == "ollama":
        from attestq.adapters import OllamaChat, OllamaEmbedder
        return Engine(chat=OllamaChat(model="llama3.1"), embed=OllamaEmbedder()), "live"
    return Engine(chat=_placeholder_chat, embed=HashEmbedder()), "demo"


@app.route("/")
def index():
    qn = demo_questionnaire()
    return render_template("index.html", questionnaire=qn, documents=DEMO_DOCUMENTS,
                           answers=None, summary=None, mode=None)


@app.route("/run", methods=["POST"])
def run():
    engine, mode = build_engine()
    qn = demo_questionnaire()
    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    answers = engine.evaluate_all(qn, namespace=DEMO_NAMESPACE)
    from attestq import summarize
    return render_template("index.html", questionnaire=qn, documents=DEMO_DOCUMENTS,
                           answers=answers, summary=summarize(answers), mode=mode,
                           markdown=to_markdown(answers, qn))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
