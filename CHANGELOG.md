# Changelog

## 0.4.0

Added the feedback layer. 0.3.0 checked a draft against its evidence; this
checks your whole pipeline against the reviewers using it. A reviewer editing a
draft before shipping is the most honest quality label a system produces, and it
costs nothing — but only if you keep both sides instead of overwriting the draft
in place.

- `attestq.feedback` — `classify_edit(draft, final)` grades a pair from
  `accepted` through `light_edit` / `heavy_edit` to `replaced`, on normalized
  text so reformatting isn't read as a correction. An item with no final answer
  is `unreviewed`, never a rejection, so acceptance rate doesn't move with queue
  depth.
- `calibration_by_confidence` + `is_calibrated` — bucket reviewed outcomes by
  confidence and test whether acceptance actually climbs with it. Returns None
  when the data is too thin for a verdict rather than calling a score
  miscalibrated off a handful of edits. `gate_check` is the low-volume version:
  pool everything either side of one threshold.
- `source_trust` — per-document acceptance, worst first, so a source that keeps
  backing rewritten answers is visible. The input to source-authority weighting.
- `build_scorecard` — the whole picture as one dict, including the SME-routing
  rate: the share of items the gate refused to draft, which is the throughput
  ceiling a daily user actually feels.
- `outcome_from_answer(answer, question, final=...)` — bridges a drafted
  `Answer` into a scoreable record, mapping citations across and recording a
  gated answer as routed to an SME. Duck-typed, so records from a system that
  never touched attestq fold into the same scorecard.

Pure stdlib and no intra-package imports, like `grounding` and `quality` — you
own the storage, this owns the arithmetic. Nothing in 0.3.0 or earlier changed.

## 0.3.0

Added the verification layer: attestq no longer stops at the draft, it checks
it. Two deterministic reports per answer, plus an optional model-backed hook for
the one call measurement can't make. Nothing here rewrites an answer — the
checks flag, and a reviewer decides.

- `Engine(verify=True)` — populates `Answer.grounding` and `Answer.quality`, and
  `Answer.needs_review` as the one-line triage signal. Off by default; it costs
  one extra embedding call per answer. A gated ("insufficient evidence") answer
  is never verified, and `None` reports mean the check did not run — which is
  deliberately not the same as a pass.
- `attestq.grounding` — extracts the concrete values a draft asserts (dates,
  versions, percentages, standards, durations) and confirms each occurs in the
  evidence, the question, or a supplied past answer. `check_answer`,
  `extract_specifics`, `find_verbatim_span`, plus annotate/parse/strip helpers
  for stamping the finding into an existing notes field without a schema change.
- `attestq.quality` — per-sentence *support* vs *echo* scoring that catches an
  answer restating its question while citing evidence it never drew on. Judged
  against the question's own similarity to its evidence, so there is no absolute
  cosine threshold to recalibrate when the embedder changes. Falls back to
  lexical scoring when no embedder is supplied or one raises, and says which
  path it took in `report.method`.
- `attestq.claim_classifier` — `make_claim_classifier(chat)` builds a
  `ClaimClassifier` over any `ChatFn` for questions that want no documentary
  evidence ("What is your registered legal entity name?"). It can only ever
  clear a flag, is memoized per question, and fails closed: an unreachable or
  unparseable model leaves the flag standing.

Both report types are plain dataclasses, so they serialize through
`answers_to_dict` / `to_json` with no extra work. Fully backward compatible —
`verify` defaults to False and `Answer` gained only optional fields.

## 0.2.0

Added finer control over the confidence gate, so a reranker can sharpen context
selection without disturbing a gate threshold calibrated to the embedder's
similarity scale.

- `Engine(gate_on=...)` — gate on the `"retrieval"` score (default; best raw
  similarity before reranking) or the `"rerank"` score (top score after
  reranking). The two coincide when no reranker is set.
- `Engine(insufficient_determination=...)` — override the determination used when
  the gate fires (e.g. `"Not Met"` instead of the last choice `"Not Applicable"`).
- `Engine(insufficient_summary=...)` — override the summary text used when the
  gate fires.

All changes are backward compatible for the common (no-reranker) case.

## 0.1.0

Initial release: model-/embedder-/store-agnostic RAG kernel for answering
security questionnaires and compliance attestations from an evidence corpus.
Core engine, in-memory + Chroma stores, OpenAI/Ollama adapters, cross-encoder
reranker, document loaders, JSON/Markdown/Word export, CLI, and web demo.
