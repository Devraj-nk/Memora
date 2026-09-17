from datetime import UTC, datetime
from pathlib import Path

import pytest

from memora.ingestion.chunker import Chunk
from memora.memory.vector_store import VectorStore


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(text=text, source="s.txt", chunk_index=index, start_offset=0, end_offset=len(text))


def test_search_on_empty_store_returns_no_matches(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    assert store.search([0.1, 0.2, 0.3]) == []


def test_add_then_search_returns_nearest_match(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    chunks = [_chunk("close match", 0), _chunk("far match", 1)]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.add(chunks, embeddings)

    results = store.search([0.9, 0.1, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].text == "close match"
    assert results[0].source == "s.txt"
    assert results[0].distance < 1.0


def test_add_across_multiple_calls_accumulates_rows(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    store.add([_chunk("first", 0)], [[1.0, 0.0, 0.0]])
    store.add([_chunk("second", 1)], [[0.0, 1.0, 0.0]])

    results = store.search([0.5, 0.5, 0.0], top_k=2)

    assert {r.text for r in results} == {"first", "second"}


def test_add_empty_list_is_a_noop(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    store.add([], [])

    assert store.search([0.1, 0.2, 0.3]) == []


def test_add_mismatched_lengths_raises(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    with pytest.raises(ValueError):
        store.add([_chunk("only one", 0)], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_all_chunks_on_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    assert store.all_chunks() == []


def test_all_chunks_returns_every_stored_chunk(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    store.add([_chunk("first", 0), _chunk("second", 1)], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    chunks = store.all_chunks()

    assert {c.text for c in chunks} == {"first", "second"}
    assert all(c.source == "s.txt" for c in chunks)


def test_all_chunks_without_modified_at_round_trips_as_none(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    store.add([_chunk("first", 0)], [[1.0, 0.0, 0.0]])

    assert store.all_chunks()[0].modified_at is None


def test_all_chunks_preserves_modified_at(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    timestamp = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    chunk = Chunk(text="a", source="s.txt", chunk_index=0, start_offset=0, end_offset=1, modified_at=timestamp)
    store.add([chunk], [[1.0, 0.0, 0.0]])

    assert store.all_chunks()[0].modified_at == timestamp


def test_all_chunks_with_vectors_on_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    assert store.all_chunks_with_vectors() == []


def test_all_chunks_with_vectors_returns_chunk_and_its_embedding(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    store.add([_chunk("first", 0)], [[1.0, 0.0, 0.0]])

    [(chunk, vector)] = store.all_chunks_with_vectors()

    assert chunk.text == "first"
    assert list(vector) == pytest.approx([1.0, 0.0, 0.0])
