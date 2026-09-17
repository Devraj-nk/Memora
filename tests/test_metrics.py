from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import memora.observability.metrics as metrics_module
from memora.ingestion.chunker import Chunk
from memora.ingestion.embedder import embed
from memora.memory.vector_store import VectorStore
from memora.observability.metrics import Metrics, compute_metrics
from memora.observability.store import ObservabilityStore, Trace


def _trace(trace_id: str, reranked: list[dict], conflicts: list[dict], context: str, response: str = "a") -> Trace:
    return Trace(
        trace_id=trace_id,
        query="q",
        retrieved=[],
        reranked=reranked,
        conflicts=conflicts,
        context=context,
        response=response,
        created_at="2026-09-16T00:00:00+00:00",
    )


@pytest.fixture(autouse=True)
def _stub_grounding(monkeypatch: pytest.MonkeyPatch):
    # is_grounded loads a real NLI model; stub it so these tests stay fast
    # and don't depend on that model being downloaded.
    monkeypatch.setattr(metrics_module, "is_grounded", lambda context, query, response: bool(context))


def test_compute_metrics_on_empty_stores_returns_zeros(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    vector_store = VectorStore(str(tmp_path / "db"))

    metrics = compute_metrics(store, vector_store)

    assert metrics == Metrics(
        trace_count=0,
        avg_sources_per_trace=0.0,
        avg_context_tokens=0.0,
        contradiction_rate=0.0,
        unique_sources_cited=0,
        answer_grounding_rate=0.0,
        avg_memory_age_days=None,
        duplicate_chunk_pairs=0,
    )


def test_compute_metrics_aggregates_trace_data(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    vector_store = VectorStore(str(tmp_path / "db"))
    store.save(_trace("t1", reranked=[{"source": "a.txt"}, {"source": "b.txt"}], conflicts=[], context="12345678"))
    store.save(
        _trace(
            "t2",
            reranked=[{"source": "a.txt"}],
            conflicts=[{"subject": "Alice", "relation": "lives in", "objects": ["Paris", "Tokyo"]}],
            context="",  # empty context -> stubbed is_grounded treats as ungrounded
        )
    )

    metrics = compute_metrics(store, vector_store)

    assert metrics.trace_count == 2
    assert metrics.avg_sources_per_trace == 1.5
    assert metrics.avg_context_tokens == 1.5  # (8//4 + 0) / 2, min 1 each -> (2+1)/2
    assert metrics.contradiction_rate == 0.5
    assert metrics.unique_sources_cited == 2
    assert metrics.answer_grounding_rate == 0.5


def test_compute_metrics_avg_memory_age_ignores_chunks_without_timestamps(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    vector_store = VectorStore(str(tmp_path / "db"))
    old = datetime.now(UTC) - timedelta(days=10)
    chunks = [
        Chunk(text="a", source="s.txt", chunk_index=0, start_offset=0, end_offset=1, modified_at=old),
        Chunk(text="b", source="s.txt", chunk_index=1, start_offset=0, end_offset=1, modified_at=None),
    ]
    vector_store.add(chunks, embed(["a", "b"]))

    metrics = compute_metrics(store, vector_store)

    assert metrics.avg_memory_age_days is not None
    assert 9.9 <= metrics.avg_memory_age_days <= 10.1


def test_compute_metrics_counts_duplicate_chunk_pairs(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    vector_store = VectorStore(str(tmp_path / "db"))
    texts = ["Python is a popular programming language.", "Python is a popular programming language."]
    chunks = [
        Chunk(text=t, source="s.txt", chunk_index=i, start_offset=0, end_offset=len(t)) for i, t in enumerate(texts)
    ]
    vector_store.add(chunks, embed(texts))

    metrics = compute_metrics(store, vector_store)

    assert metrics.duplicate_chunk_pairs == 1
