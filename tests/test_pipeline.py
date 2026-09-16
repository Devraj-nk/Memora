from pathlib import Path

import pytest

import memora.ingestion.pipeline as pipeline_module
from memora.ingestion.embedder import embed
from memora.ingestion.pipeline import ingest_source
from memora.knowledge_graph.extractor import Relationship
from memora.knowledge_graph.graph_store import GraphStore
from memora.memory.vector_store import VectorStore


def test_ingest_source_adds_chunks_to_store(tmp_path: Path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("Memora tracks how memory is retrieved and used.", encoding="utf-8")
    store = VectorStore(str(tmp_path / "db"))

    result = ingest_source(str(file_path), store)

    assert result.source == str(file_path)
    assert result.chunk_count == 1

    matches = store.search(embed(["memory retrieval"])[0], top_k=1)
    assert matches[0].source == str(file_path)


def test_ingest_source_empty_file_returns_zero_chunks_and_skips_store(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ", encoding="utf-8")
    store = VectorStore(str(tmp_path / "db"))

    result = ingest_source(str(file_path), store)

    assert result.chunk_count == 0
    assert store.search(embed(["anything"])[0]) == []


def test_ingest_source_extracts_relationships_into_graph_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("Memora uses LanceDB for vector search.", encoding="utf-8")
    store = VectorStore(str(tmp_path / "db"))
    graph_store = GraphStore(str(tmp_path / "graph.sqlite3"))
    monkeypatch.setattr(
        pipeline_module,
        "extract",
        lambda text: [Relationship(subject="Memora", relation="uses", object="LanceDB")],
    )

    ingest_source(str(file_path), store, graph_store)

    assert graph_store.related_chunks("Memora") == [(str(file_path), 0)]


def test_ingest_source_without_graph_store_skips_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("Memora uses LanceDB for vector search.", encoding="utf-8")
    store = VectorStore(str(tmp_path / "db"))
    called: list[str] = []
    monkeypatch.setattr(pipeline_module, "extract", lambda text: called.append(text) or [])

    ingest_source(str(file_path), store)

    assert called == []
