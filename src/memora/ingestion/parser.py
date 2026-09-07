"""Turns a raw source (file path) into plain text + metadata.

Plain text/markdown/code first; PDF/docx/OCR are handled later via the
optional `parsing` dependency group (see pyproject.toml).
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Extensions read directly as UTF-8 text. Binary formats (PDF, docx, images)
# need the optional `parsing` extra and aren't handled here yet.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".sh", ".sql",
}

# mimetypes' guesses for these vary by OS/Python version; pin them so
# behavior is deterministic.
_MIME_OVERRIDES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    source: str
    mime_type: str
    modified_at: datetime


def parse(source_path: str) -> ParsedDocument:
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(source_path)

    suffix = path.suffix.lower()
    if suffix not in _TEXT_EXTENSIONS:
        raise ValueError(
            f"Unsupported source type {suffix!r} for {source_path!r}; "
            "install the 'parsing' extra for PDF/docx/OCR support."
        )

    text = path.read_text(encoding="utf-8", errors="replace")
    mime_type = _MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(path.name)[0] or "text/plain"
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    return ParsedDocument(
        text=text,
        source=str(path),
        mime_type=mime_type,
        modified_at=modified_at,
    )
