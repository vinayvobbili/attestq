# attestq

**Answer security questionnaires and compliance attestations from your own evidence.**

`attestq` is a small, model-agnostic RAG kernel for a problem every security and
GRC team has: you have a *questionnaire* (a vendor security review, a SIG/CAIQ
response, an audit-evidence request, a due-diligence form, the security section
of an RFP) and you have a *pile of evidence* (SOC 2 reports, policies, standards,
prior questionnaires). `attestq` retrieves the relevant evidence for each
question and drafts a grounded, **cited** answer — and, just as importantly, tells
you plainly when the evidence isn't there.

It is **not** a classic ML / training problem. It's retrieval + an LLM you bring
yourself. attestq owns the orchestration; you own the model, the embedder, and
the store.

## Why it exists

The same pattern keeps getting rebuilt one-off inside every program: chunk the
docs, embed them, retrieve per question, prompt an LLM, paste into a form. Done
naively it hallucinates (answers with no supporting evidence) and it silently
drops the one focused document that actually had the answer. attestq bakes in the
two hard-won fixes:

- **Confidence gate.** When the best retrieved evidence scores below a threshold,
  the question is answered *"insufficient evidence"* **without calling the LLM**.
  Absence of evidence is a first-class, valid result — not an invitation to guess.
- **Wide rerank window.** The kernel keeps a generous number of chunks after
  reranking so a single relevant document isn't dropped on a small corpus.

## Install

```bash
pip install attestq                 # dependency-free core
pip install "attestq[chroma]"       # + persistent Chroma vector store
pip install "attestq[openai]"       # + OpenAI-compatible chat adapter
pip install "attestq[ollama]"       # + local Ollama embeddings
pip install "attestq[rerank]"       # + cross-encoder reranker
pip install "attestq[loaders]"      # + pdf / docx / xlsx loaders
pip install "attestq[all]"          # everything
```

The core has **zero** third-party dependencies. Adapters are opt-in extras.

## Quick start

You inject any chat model and any embedder as plain callables:

```python
from attestq import Engine, Question

# Bring your own model + embedder (one-liners around any provider).
def my_chat(prompt: str) -> str:
    ...   # call OpenAI, Anthropic, a local model, your corporate gateway, ...

def my_embed(texts):
    ...   # return one vector per text

engine = Engine(chat=my_chat, embed=my_embed)   # in-memory store by default

# Ingest a vendor's evidence into its own namespace.
engine.ingest(
    [
        ("All customer data at rest is encrypted with AES-256...", {"source": "DataProtection.pdf"}),
        ("MFA is enforced for all privileged access...", {"source": "AccessControl.docx"}),
    ],
    namespace="helios",
)

# Answer one control.
ans = engine.evaluate(
    Question(
        id="ENC-1",
        prompt="Is customer data encrypted at rest?",
        choices=["Met", "Not Met", "Not Applicable"],
    ),
    namespace="helios",
)

print(ans.determination)          # "Met"
print(ans.confidence)             # 0.0 - 1.0 retrieval confidence
print(ans.insufficient_evidence)  # False
for c in ans.citations:
    print(c.source, "->", c.snippet)
```

Run a whole questionnaire:

```python
from attestq import Questionnaire

qn = Questionnaire(
    id="vendor-sec-review",
    title="Vendor Security Review",
    questions=[
        Question(id="ENC-1", prompt="Is data encrypted at rest?", choices=["Met", "Not Met", "Not Applicable"]),
        Question(id="IAM-1", prompt="Is MFA enforced for privileged access?", choices=["Met", "Not Met", "Not Applicable"]),
    ],
)

answers = engine.evaluate_all(qn, namespace="helios", on_answer=lambda a: print(a.question_id, a.determination))
```

## How it works

```
evidence docs ──▶ chunk ──▶ embed ──▶ vector store (namespaced per corpus)

question ──▶ embed ──▶ retrieve(k) ──▶ rerank(top_k) ──▶ confidence gate
                                                              │
                              below threshold ───────────────┤──▶ "insufficient evidence" (no LLM call)
                                                              │
                              above threshold ──▶ prompt LLM ─┴──▶ parse ──▶ Answer(determination, summary, citations, confidence)
```

Everything is swappable:

| Piece | Default | Swap for |
|-------|---------|----------|
| Chat model | *you inject it* | any LLM / gateway |
| Embedder | *you inject it* | Ollama, sentence-transformers, OpenAI |
| Vector store | `InMemoryVectorStore` | `attestq[chroma]` |
| Reranker | none | `attestq[rerank]` cross-encoder |
| Prompt / parser | evidence-only default | your own `prompt_builder` / `response_parser` |

## Design principles

- **Bring your own model.** No provider is hard-wired. A lambda is enough.
- **Grounded or silent.** Answers cite their evidence; thin evidence yields an
  explicit "insufficient" result, never a confident guess.
- **Per-corpus isolation.** One store, many namespaces — keep each vendor's
  evidence separate without standing up a new index each time.
- **Light core.** The kernel imports nothing third-party; heavy deps stay in
  extras you opt into.

## Status

Early but usable. Core kernel + in-memory store are here today; provider adapters,
document loaders, and export renderers land as optional extras. Contributions and
issues welcome.

## License

MIT
