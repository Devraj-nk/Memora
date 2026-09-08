from datetime import datetime, timezone

import pytest

from memora.ingestion.chunker import chunk
from memora.ingestion.parser import ParsedDocument


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        source="test.md",
        mime_type="text/markdown",
        modified_at=datetime.now(timezone.utc),
    )


def test_chunk_short_text_returns_single_chunk() -> None:
    doc = _doc("Just a short note.")

    chunks = chunk(doc, chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == "Just a short note."
    assert chunks[0].source == "test.md"
    assert chunks[0].chunk_index == 0


def test_chunk_splits_long_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    doc = _doc(text)

    chunks = chunk(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert chunks[0].text.startswith("word0")
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].chunk_index == chunks[i].chunk_index + 1
        # consecutive chunks overlap
        assert chunks[i].end_offset > chunks[i + 1].start_offset


def test_chunk_empty_text_returns_no_chunks() -> None:
    assert chunk(_doc("   \n  ")) == []


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk(_doc("some text"), chunk_size=100, chunk_overlap=100)
