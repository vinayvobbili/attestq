# Changelog

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
