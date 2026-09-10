from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memora.api.main import app
from memora.api.routes.ingest import get_vector_store
from memora.memory.vector_store import VectorStore


@pytest.fixture
def client(tmp_path: Path):
    app.dependency_overrides[get_vector_store] = lambda: VectorStore(str(tmp_path / "db"))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_ingest_endpoint_returns_chunk_count(client: TestClient, tmp_path: Path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("Some note content for ingestion.", encoding="utf-8")

    response = client.post("/ingest/", json={"path": str(file_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == str(file_path)
    assert body["chunk_count"] == 1


def test_ingest_endpoint_missing_file_returns_404(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/ingest/", json={"path": str(tmp_path / "missing.md")})

    assert response.status_code == 404


def test_ingest_endpoint_unsupported_extension_returns_400(client: TestClient, tmp_path: Path) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    response = client.post("/ingest/", json={"path": str(file_path)})

    assert response.status_code == 400
