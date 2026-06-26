"""Zero-setup quickstart: runs anywhere, no API key, no model download.

Uses the built-in HashEmbedder for retrieval and a placeholder chat function so
you can see the shape of the API and watch the confidence gate fire for real.
Swap `placeholder_chat` for a real LLM (see README: OpenAIChat / OllamaChat) to
get reasoned determinations.

    python examples/quickstart.py
"""

from attestq import Engine, HashEmbedder, Question, to_markdown


def placeholder_chat(prompt: str) -> str:
    # A real LLM goes here. This stub just returns the expected labelled format
    # so the example runs offline; it does NOT actually reason about evidence.
    return (
        "DETERMINATION: Met\n"
        "EVIDENCE SUMMARY: (placeholder — connect a real LLM for a real rationale)\n"
        "CITATIONS: 1\n"
        "NOTES: none"
    )


def main():
    engine = Engine(chat=placeholder_chat, embed=HashEmbedder())

    engine.ingest(
        [
            ("All customer data at rest is encrypted with AES-256; TLS 1.2+ in transit.",
             {"source": "DataProtection.txt"}),
            ("Multi-factor authentication is enforced for all privileged access.",
             {"source": "AccessControl.txt"}),
        ],
        namespace="acme",
    )

    questions = [
        Question(id="ENC-1", prompt="Is customer data encrypted at rest?",
                 choices=["Met", "Not Met", "Not Applicable"], domain="Data Protection"),
        # No evidence for this one -> the confidence gate answers it without an LLM call.
        Question(id="SDLC-1", prompt="Does the vendor run SAST and DAST in its build pipeline?",
                 choices=["Met", "Not Met", "Not Applicable"], domain="Secure Development"),
    ]

    answers = engine.evaluate_all(questions, namespace="acme")
    for a in answers:
        gate = " (insufficient evidence — no LLM call)" if a.insufficient_evidence else ""
        print(f"{a.question_id}: {a.determination} [conf {a.confidence:.2f}]{gate}")

    print("\n--- Markdown report ---\n")
    print(to_markdown(answers))


if __name__ == "__main__":
    main()
