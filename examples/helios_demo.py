"""Full sample assessment against a real LLM.

Runs the bundled fictional "Helios Data Systems" vendor through a security
due-diligence questionnaire and writes a Markdown report. Wire your provider in
`build_providers()` below — OpenAI-compatible or local Ollama.

    # OpenAI (or any compatible endpoint)
    export OPENAI_API_KEY=sk-...
    python examples/helios_demo.py

    # ...or local Ollama
    PROVIDER=ollama python examples/helios_demo.py

Equivalent one-liner once installed:  attestq demo -o report.md
"""

import os

from attestq import Engine, to_markdown
from attestq.demo import DEMO_DOCUMENTS, DEMO_NAMESPACE, demo_questionnaire


def build_providers():
    """Return (chat, embed) callables for your chosen provider."""
    provider = os.environ.get("PROVIDER", "openai" if os.environ.get("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        from attestq.adapters import OpenAIChat, OpenAIEmbedder
        return OpenAIChat(model="gpt-4o-mini"), OpenAIEmbedder(model="text-embedding-3-small")
    if provider == "ollama":
        from attestq.adapters import OllamaChat, OllamaEmbedder
        return OllamaChat(model="llama3.1"), OllamaEmbedder(model="nomic-embed-text")
    raise SystemExit(f"unknown PROVIDER={provider!r}")


def main():
    chat, embed = build_providers()
    engine = Engine(chat=chat, embed=embed)

    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    qn = demo_questionnaire()

    answers = engine.evaluate_all(
        qn,
        namespace=DEMO_NAMESPACE,
        on_answer=lambda a: print(f"  {a.question_id}: {a.determination}"),
    )

    report = to_markdown(answers, questionnaire=qn)
    with open("report.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\nWrote report.md")


if __name__ == "__main__":
    main()
