"""Command-line interface for attestq.

    attestq demo                       # run the bundled sample assessment
    attestq run -q q.yaml -e ./evidence -o report.md
    attestq serve --demo --offline     # HTTP API for the Chrome extension
    attestq version

Providers are resolved from flags or environment so the same command works
against OpenAI-compatible endpoints or a local Ollama. See ``attestq run -h``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

from . import __version__
from .demo import DEMO_DOCUMENTS, DEMO_NAMESPACE, demo_questionnaire
from .engine import Engine
from .export import summarize, to_docx, to_json, to_markdown
from .io import load_questionnaire
from .loaders import load_documents
from .models import Questionnaire


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# --- commands -----------------------------------------------------------------


def _cmd_version(args) -> int:
    print(f"attestq {__version__}")
    return 0


def _cmd_demo(args) -> int:
    engine = _build_engine(args)
    qn = demo_questionnaire()
    print(f"Ingesting {len(DEMO_DOCUMENTS)} sample evidence documents...", file=sys.stderr)
    engine.ingest(DEMO_DOCUMENTS, namespace=DEMO_NAMESPACE)
    return _run_and_emit(engine, qn, DEMO_NAMESPACE, args)


def _cmd_run(args) -> int:
    qn = load_questionnaire(args.questionnaire)
    docs = _collect_evidence(args.evidence)
    if not docs:
        print("error: no readable evidence documents found", file=sys.stderr)
        return 2
    engine = _build_engine(args)
    print(f"Ingesting {len(docs)} evidence documents...", file=sys.stderr)
    engine.ingest(docs, namespace=args.namespace)
    return _run_and_emit(engine, qn, args.namespace, args)


def _cmd_serve(args) -> int:
    from .adapters._util import require

    uvicorn = require("uvicorn", "server")
    from .server import create_app

    engine = _build_serve_engine(args)
    if args.demo:
        _ingest_once(engine, DEMO_DOCUMENTS, DEMO_NAMESPACE)
    if args.evidence:
        docs = _collect_evidence(args.evidence)
        if not docs:
            print("error: no readable evidence documents found", file=sys.stderr)
            return 2
        _ingest_once(engine, docs, args.namespace)

    token = args.token or os.environ.get("ATTESTQ_API_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("warning: listening beyond localhost with no --token; anyone who can reach "
              "this port can query your evidence", file=sys.stderr)
    app = create_app(engine, api_token=token, feedback_path=args.feedback, workers=args.workers)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _build_serve_engine(args) -> Engine:
    store = None
    if args.chroma:
        from .adapters import ChromaStore

        store = ChromaStore(path=args.chroma, collection=args.collection)
    if args.offline:
        from .embedders import HashEmbedder

        chat, embed = offline_chat, HashEmbedder()
        print("Offline mode: token-overlap retrieval and placeholder answers. "
              "Configure a provider for real ones.", file=sys.stderr)
    else:
        chat, embed = _build_providers(args)
    return Engine(chat=chat, embed=embed, store=store,
                  min_confidence=args.min_confidence, verify=args.verify)


def _ingest_once(engine: Engine, docs, namespace: str) -> None:
    """Ingest unless the namespace already has chunks — a persistent store survives
    restarts, and re-ingesting would duplicate every chunk."""
    existing = engine.store.count(namespace)
    if existing:
        print(f"'{namespace}' already holds {existing} chunks; not re-ingesting.", file=sys.stderr)
        return
    n = engine.ingest(docs, namespace=namespace)
    print(f"Ingested {n} chunks into '{namespace}'.", file=sys.stderr)


def offline_chat(prompt: str) -> str:
    """Placeholder model for trying the service with no provider.

    Picks the first allowed determination and quotes the top excerpt, so every
    moving part — retrieval, the gate, citations, a client filling a form — runs
    for real while the verdicts themselves are obviously canned.
    """
    import re

    allowed = re.search(r"DETERMINATION must be exactly one of: (.+)\.", prompt)
    determination = allowed.group(1).split(", ")[0] if allowed else "See evidence"
    excerpt = re.search(r"\[1\] \(source: [^)]*\)\n(.+)", prompt)
    quote = excerpt.group(1).strip()[:200] if excerpt else "no excerpt"
    return (
        f"DETERMINATION: {determination}\n"
        f"EVIDENCE SUMMARY: [offline placeholder] Top evidence: {quote}\n"
        "CITATIONS: 1\n"
        "NOTES: none"
    )


def _run_and_emit(engine: Engine, qn: Questionnaire, namespace: str, args) -> int:
    def progress(ans):
        flag = " [insufficient evidence]" if ans.insufficient_evidence else ""
        print(f"  {ans.question_id}: {ans.determination}{flag}", file=sys.stderr)

    print(f"Evaluating {len(qn)} questions...", file=sys.stderr)
    answers = engine.evaluate_all(qn, namespace=namespace, on_answer=progress)

    s = summarize(answers)
    print(
        f"\nDone. {s['total']} questions | "
        + ", ".join(f"{k}: {v}" for k, v in s["by_determination"].items())
        + f" | avg confidence {s['average_confidence']:.2f}",
        file=sys.stderr,
    )

    fmt = args.format or _format_from_path(args.out)
    if fmt == "docx":
        if not args.out:
            print("error: --format docx requires --out PATH", file=sys.stderr)
            return 2
        to_docx(answers, args.out, questionnaire=qn)
        print(f"Wrote {args.out}", file=sys.stderr)
        return 0

    rendered = to_json(answers, qn) if fmt == "json" else to_markdown(answers, qn)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(rendered)
    return 0


# --- provider / engine wiring -------------------------------------------------


class ProviderError(RuntimeError):
    pass


def _build_engine(args) -> Engine:
    chat, embed = _build_providers(args)
    return Engine(chat=chat, embed=embed, min_confidence=args.min_confidence)


def _build_providers(args) -> Tuple:
    provider = args.provider
    if provider == "auto":
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"

    if provider == "openai":
        try:
            from .adapters import OpenAIChat, OpenAIEmbedder
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(str(exc)) from exc
        if not os.environ.get("OPENAI_API_KEY") and not args.base_url:
            raise ProviderError(
                "openai provider needs OPENAI_API_KEY (or --base-url for a local/"
                "compatible endpoint). Try '--provider ollama' for a local model."
            )
        chat = OpenAIChat(
            model=args.chat_model or os.environ.get("ATTESTQ_CHAT_MODEL", "gpt-4o-mini"),
            base_url=args.base_url or os.environ.get("ATTESTQ_BASE_URL"),
        )
        embed = OpenAIEmbedder(
            model=args.embed_model or os.environ.get("ATTESTQ_EMBED_MODEL", "text-embedding-3-small"),
            base_url=args.base_url or os.environ.get("ATTESTQ_BASE_URL"),
        )
        return chat, embed

    if provider == "ollama":
        try:
            from .adapters import OllamaChat, OllamaEmbedder
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(str(exc)) from exc
        host = args.base_url or os.environ.get("ATTESTQ_OLLAMA_HOST", "http://localhost:11434")
        chat = OllamaChat(model=args.chat_model or os.environ.get("ATTESTQ_CHAT_MODEL", "llama3.1"), host=host)
        embed = OllamaEmbedder(model=args.embed_model or os.environ.get("ATTESTQ_EMBED_MODEL", "nomic-embed-text"), host=host)
        return chat, embed

    raise ProviderError(f"unknown provider: {provider!r}")


def _collect_evidence(paths: List[str]) -> List[dict]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                files.extend(os.path.join(root, n) for n in sorted(names))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"warning: skipping missing path {p}", file=sys.stderr)
    return load_documents(files)


def _format_from_path(path) -> str:
    if not path:
        return "md"
    ext = os.path.splitext(path)[1].lower()
    return {".json": "json", ".docx": "docx", ".md": "md", ".markdown": "md"}.get(ext, "md")


# --- argument parser ----------------------------------------------------------


def _add_provider_args(p: argparse.ArgumentParser, report: bool = True) -> None:
    p.add_argument("--provider", choices=["auto", "openai", "ollama"], default="auto",
                   help="LLM/embedding provider (default: auto - openai if OPENAI_API_KEY else ollama)")
    p.add_argument("--chat-model", help="chat model name override")
    p.add_argument("--embed-model", help="embedding model name override")
    p.add_argument("--base-url", help="base URL for an OpenAI-compatible endpoint, or Ollama host")
    p.add_argument("--min-confidence", type=float, default=0.45,
                   help="retrieval-score gate; below this -> insufficient evidence (default: 0.45)")
    if not report:
        return
    p.add_argument("-o", "--out", help="write the report to this path (default: stdout)")
    p.add_argument("--format", choices=["md", "json", "docx"], help="output format (default: inferred from --out, else md)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attestq",
        description="Answer security questionnaires and compliance attestations from your evidence.",
    )
    sub = parser.add_subparsers()

    p_demo = sub.add_parser("demo", help="run the bundled sample assessment")
    _add_provider_args(p_demo)
    p_demo.set_defaults(func=_cmd_demo)

    p_run = sub.add_parser("run", help="evaluate a questionnaire against evidence")
    p_run.add_argument("-q", "--questionnaire", required=True, help="questionnaire .json/.yaml file")
    p_run.add_argument("-e", "--evidence", required=True, nargs="+", help="evidence files and/or directories")
    p_run.add_argument("-n", "--namespace", default="default", help="corpus namespace (default: default)")
    _add_provider_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_serve = sub.add_parser("serve", help="serve an HTTP API for browser clients (needs attestq[server])")
    p_serve.add_argument("--demo", action="store_true", help=f"load the bundled sample evidence as '{DEMO_NAMESPACE}'")
    p_serve.add_argument("--offline", action="store_true",
                         help="no provider: token-overlap retrieval and placeholder answers, for trying it out")
    p_serve.add_argument("-e", "--evidence", nargs="+", help="evidence files/directories to ingest at startup")
    p_serve.add_argument("-n", "--namespace", default="default", help="namespace for --evidence (default: default)")
    p_serve.add_argument("--chroma", metavar="PATH", help="persistent Chroma store directory (default: in memory)")
    p_serve.add_argument("--collection", default="attestq_evidence", help="Chroma collection name")
    p_serve.add_argument("--verify", action="store_true", help="run grounding/quality checks on every answer")
    p_serve.add_argument("--feedback", metavar="PATH", help="JSONL file for reviewer feedback; enables /feedback and /scorecard")
    p_serve.add_argument("--token", help="require this bearer token (default: $ATTESTQ_API_TOKEN)")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    p_serve.add_argument("--workers", type=int, default=8, help="questions answered in parallel (default: 8)")
    _add_provider_args(p_serve, report=False)
    p_serve.set_defaults(func=_cmd_serve)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=_cmd_version)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
