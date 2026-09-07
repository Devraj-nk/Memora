from pathlib import Path

import pytest

from memora.ingestion.parser import ParsedDocument, parse


def test_parse_reads_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("# Hello\n\nSome content.", encoding="utf-8")

    doc = parse(str(file_path))

    assert isinstance(doc, ParsedDocument)
    assert doc.text == "# Hello\n\nSome content."
    assert doc.source == str(file_path)
    assert doc.mime_type == "text/markdown"


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse(str(tmp_path / "missing.md"))


def test_parse_unsupported_extension_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError):
        parse(str(file_path))
