"""A small, dependency-free recursive text splitter.

Good enough for evidence documents out of the box. If you already chunk upstream
(LangChain, unstructured, etc.), pass pre-split text to Engine.ingest and skip
this entirely.
"""

from __future__ import annotations

from typing import List

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
    separators: List[str] | None = None,
) -> List[str]:
    """Split `text` into overlapping chunks, preferring natural boundaries.

    Tries each separator in turn, falling back to harder splits only when a piece
    is still too large. Overlap preserves context across chunk edges so a fact
    that straddles a boundary is still retrievable.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = separators or _DEFAULT_SEPARATORS
    pieces = _recursive_split(text, chunk_size, separators)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        # Hard character split as a last resort.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        piece = part + sep
        if len(piece) <= chunk_size:
            out.append(piece)
        elif rest:
            out.extend(_recursive_split(part, chunk_size, rest))
        else:
            out.append(piece)
    return out


def _merge_with_overlap(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = (current[-overlap:] if overlap else "") + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]
