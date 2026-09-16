from pathlib import Path

from memora.observability.metrics import Metrics, compute_metrics
from memora.observability.store import ObservabilityStore, Trace


def _trace(trace_id: str, reranked: list[dict], conflicts: list[dict], context: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        query="q",
        retrieved=[],
        reranked=reranked,
        conflicts=conflicts,
        context=context,
        response="a",
        created_at="2026-09-16T00:00:00+00:00",
    )


def test_compute_metrics_on_empty_store_returns_zeros(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))

    assert compute_metrics(store) == Metrics(0, 0.0, 0.0, 0.0, 0)


def test_compute_metrics_aggregates_across_traces(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    store.save(_trace("t1", reranked=[{"source": "a.txt"}, {"source": "b.txt"}], conflicts=[], context="12345678"))
    store.save(
        _trace(
            "t2",
            reranked=[{"source": "a.txt"}],
            conflicts=[{"subject": "Alice", "relation": "lives in", "objects": ["Paris", "Tokyo"]}],
            context="1234",
        )
    )

    metrics = compute_metrics(store)

    assert metrics.trace_count == 2
    assert metrics.avg_sources_per_trace == 1.5
    assert metrics.avg_context_tokens == 1.5
    assert metrics.contradiction_rate == 0.5
    assert metrics.unique_sources_cited == 2
