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

## Command line

Installing attestq gives you an `attestq` command. Run the bundled sample, or
point it at your own questionnaire and evidence:

```bash
# Run the bundled fictional sample assessment end-to-end
attestq demo -o report.md

# Evaluate your questionnaire against a folder of evidence
attestq run -q questionnaire.yaml -e ./vendor-evidence -n acme -o report.docx

# Pipe JSON to another tool
attestq run -q q.json -e ./evidence --format json | jq .summary
```

Providers are resolved from flags or environment, so the same command works
against OpenAI-compatible endpoints or a local Ollama:

```bash
export OPENAI_API_KEY=sk-...                 # uses OpenAI by default
attestq demo

attestq demo --provider ollama               # local, no key, nothing leaves the host
attestq run -q q.yaml -e ./ev --provider openai --base-url https://my-gateway/v1
```

## Try it instantly (no setup)

The built-in `HashEmbedder` needs no model and no service, so you can watch the
retrieval pipeline and the confidence gate work the moment you install:

```bash
pip install attestq
python examples/quickstart.py
```

See [`examples/`](examples/) for a full provider-wired demo
(`helios_demo.py`) and a minimal web UI (`examples/web/`) that runs the sample
assessment in your browser.

## How it works

```
evidence docs ──▶ chunk ──▶ embed ──▶ vector store (namespaced per corpus)

question ──▶ embed ──▶ retrieve(k) ──▶ rerank(top_k) ──▶ confidence gate
                                                              │
                              below threshold ───────────────┤──▶ "insufficient evidence" (no LLM call)
                                                              │
                              above threshold ──▶ prompt LLM ─┴──▶ parse ──▶ Answer(determination, summary, citations, confidence)
```

## Verifying the draft

Drafting is half the job; the other half is not trusting the draft. Turn on
`verify=True` and every answer carries two reports:

```python
engine = Engine(chat=my_llm, embed=my_embedder, verify=True)
answer = engine.evaluate(question, namespace="vendor-x")

if answer.needs_review:
    print(answer.grounding.unverified)   # ['2019-03-01'] — asserted, not in evidence
    print(answer.quality.detail())       # "only 20% of the answer is carried by ..."
```

**`answer.grounding`** extracts every concrete value the draft asserted — dates,
versions, percentages, standards, durations — and confirms each one actually
occurs in the evidence. This catches the highest-liability failure: *"certified
to ISO 27001 since 2019-03-01"* when neither appears in a single retrieved
document.

**`answer.quality`** catches the quieter failure — an answer that restates the
question and cites evidence it never drew on:

> **Q:** Are encryption keys securely managed through defined key-management processes?
> **A:** Yes. Encryption keys are securely managed through defined key-management processes.

That asserts nothing the question didn't already say. Per sentence, it measures
*support* (closeness to the retrieved evidence) against *echo* (closeness to the
question), each judged against the question's own similarity to that evidence —
so there is no magic cosine constant to recalibrate when you swap embedders.

Both checks are **deterministic**: no second LLM grading the first, which would
only share the drafter's blind spots. Both **flag and never rewrite** — a
flagged answer may still be the right answer, so the disposition stays with the
reviewer. Verification is off by default; it costs one extra embedding call per
answer.

One judgment measurement genuinely can't make is whether a question wanted
documentary evidence at all — *"What is your registered legal entity name?"* has
a correct answer no policy document will ever back. Inject a small model to
decide, and it may only ever **clear** a flag, never raise one:

```python
from attestq import make_claim_classifier

engine = Engine(chat=big_model, embed=embedder, verify=True,
                claim_classifier=make_claim_classifier(small_fast_model))
```

Every failure path in that classifier — endpoint down, unparseable reply —
leaves the flag standing.

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
- **The model proposes, deterministic code disposes.** Verification is plain
  string and vector math, so the verifier cannot hallucinate the way an
  LLM-grading-an-LLM judge can. It flags for a human; it never rewrites.
- **Per-corpus isolation.** One store, many namespaces — keep each vendor's
  evidence separate without standing up a new index each time.
- **Light core.** The kernel imports nothing third-party; heavy deps stay in
  extras you opt into.

## Status

Usable today: the core kernel, the verification layer, in-memory + Chroma
stores, OpenAI/Ollama adapters, a cross-encoder reranker, document loaders,
JSON/Markdown/Word export, a CLI, and a web demo. Contributions and issues
welcome.

## License

MIT
