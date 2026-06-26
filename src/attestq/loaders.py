"""Document loaders: extract plain text from evidence files.

Turns the file formats real evidence arrives in — PDF, Word, Excel, plain text /
markdown — into text ready for ``Engine.ingest``. PDF/DOCX/XLSX support needs the
optional ``attestq[loaders]`` extra; plain text and markdown need nothing.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

from .adapters._util import require

_TEXT_EXTS = {".txt", ".md", ".markdown", ".text", ".csv", ".log", ".rst", ".json"}


def load_document(path: str) -> str:
    """Extract text from a single file, dispatching on its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in (".docx",):
        return _load_docx(path)
    if ext in (".xlsx", ".xlsm"):
        return _load_xlsx(path)
    if ext in _TEXT_EXTS or ext == "":
        return _load_text(path)
    # Unknown extension: try plain text rather than failing outright.
    return _load_text(path)


def load_documents(paths: Iterable[str]) -> List[Dict]:
    """Load many files into ``Engine.ingest``-ready dicts.

    Each result is ``{"text", "source", "filename"}`` — pass the list straight to
    ``engine.ingest(...)``. Empty/unreadable files are skipped.
    """
    docs: List[Dict] = []
    for path in paths:
        text = load_document(path)
        if text and text.strip():
            name = os.path.basename(path)
            docs.append({"text": text, "source": name, "filename": name})
    return docs


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _load_pdf(path: str) -> str:
    pypdf = require("pypdf", "loaders")
    reader = pypdf.PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _load_docx(path: str) -> str:
    docx = require("docx", "loaders")
    document = docx.Document(path)
    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_xlsx(path: str) -> str:
    openpyxl = require("openpyxl", "loaders")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts: List[str] = []
    for ws in wb.worksheets:
        parts.append(f"# Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append(" | ".join(cells))
    wb.close()
    return "\n".join(parts)
