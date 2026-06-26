"""Shared helpers for optional-dependency adapters."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(import_name: str, extra: str) -> ModuleType:
    """Import an optional dependency or raise a friendly install hint.

    Args:
        import_name: The importable module name (e.g. "chromadb").
        extra: The attestq extra that provides it (e.g. "chroma").
    """
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:  # pragma: no cover - exercised via adapters
        raise ImportError(
            f"This adapter needs the optional dependency '{import_name}'. "
            f"Install it with:  pip install \"attestq[{extra}]\""
        ) from exc


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid mapping any real to (0, 1)."""
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
