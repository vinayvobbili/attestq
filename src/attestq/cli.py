"""Command-line interface for attestq.

    attestq demo                       # run the bundled sample assessment
    attestq run -q q.yaml -e ./evidence -o report.md
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


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=["auto", "openai", "ollama"], default="auto",
                   help="LLM/embedding provider (default: auto - openai if OPENAI_API_KEY else ollama)")
    p.add_argument("--chat-model", help="chat model name override")
    p.add_argument("--embed-model", help="embedding model name override")
    p.add_argument("--base-url", help="base URL for an OpenAI-compatible endpoint, or Ollama host")
    p.add_argument("--min-confidence", type=float, default=0.45,
                   help="retrieval-score gate; below this -> insufficient evidence (default: 0.45)")
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

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=_cmd_version)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
