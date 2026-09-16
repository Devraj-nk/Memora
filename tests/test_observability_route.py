from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memora.api.dependencies import get_observability_store
from memora.api.main import app
from memora.observability.store import ObservabilityStore, Trace


@pytest.fixture
def observability_store(tmp_path: Path) -> ObservabilityStore:
    return ObservabilityStore(str(tmp_path / "obs.sqlite3"))


@pytest.fixture
def client(observability_store: ObservabilityStore):
    app.dependency_overrides[get_observability_store] = lambda: observability_store
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_get_trace_returns_saved_trace(client: TestClient, observability_store: ObservabilityStore) -> None:
    observability_store.save(
        Trace(
            trace_id="t1",
            query="q",
            retrieved=[],
            reranked=[],
            conflicts=[],
            context="ctx",
            response="the answer",
            created_at="2026-09-16T00:00:00+00:00",
        )
    )

    response = client.get("/observability/traces/t1")

    assert response.status_code == 200
    assert response.json()["response"] == "the answer"


def test_get_trace_missing_returns_404(client: TestClient) -> None:
    response = client.get("/observability/traces/missing")

    assert response.status_code == 404


def test_get_metrics_on_empty_store(client: TestClient) -> None:
    response = client.get("/observability/metrics")

    assert response.status_code == 200
    assert response.json()["trace_count"] == 0
