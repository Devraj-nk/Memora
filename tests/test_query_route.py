from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import memora.retrieval.router as router_module
from memora.api.dependencies import get_observability_store, get_vector_store
from memora.api.main import app
from memora.ingestion.pipeline import ingest_source
from memora.memory.vector_store import VectorStore
from memora.observability.store import ObservabilityStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(str(tmp_path / "db"))


@pytest.fixture
def observability_store(tmp_path: Path) -> ObservabilityStore:
    return ObservabilityStore(str(tmp_path / "obs.sqlite3"))


@pytest.fixture
def client(store: VectorStore, observability_store: ObservabilityStore, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(router_module, "generate", lambda prompt: "a generated answer")
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_observability_store] = lambda: observability_store
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_on_empty_store_returns_no_sources(client: TestClient) -> None:
    response = client.post("/query/", json={"query": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "a generated answer"
    assert body["sources"] == []
    assert body["trace_id"]


def test_query_endpoint_returns_ingested_content_as_a_source(
    client: TestClient, store: VectorStore, tmp_path: Path
) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("Memora tracks how memory is retrieved and used.", encoding="utf-8")
    ingest_source(str(file_path), store)

    response = client.post("/query/", json={"query": "memory retrieval"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == str(file_path)


def test_query_endpoint_records_a_trace(
    client: TestClient, observability_store: ObservabilityStore, tmp_path: Path
) -> None:
    response = client.post("/query/", json={"query": "hello"})

    trace_id = response.json()["trace_id"]
    saved = observability_store.get(trace_id)
    assert saved is not None
    assert saved.query == "hello"
    assert saved.response == "a generated answer"
