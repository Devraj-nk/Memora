from pathlib import Path

from memora.observability.store import ObservabilityStore, Trace


def _trace(trace_id: str = "t1") -> Trace:
    return Trace(
        trace_id=trace_id,
        query="what is memora?",
        retrieved=[{"text": "a"}],
        reranked=[{"text": "a", "score": 0.9}],
        context="[s.txt]\na",
        response="Memora is a memory engine.",
        created_at="2026-09-13T00:00:00+00:00",
    )


def test_get_missing_trace_returns_none(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))

    assert store.get("missing") is None


def test_save_then_get_roundtrips_a_trace(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    trace = _trace()

    store.save(trace)

    assert store.get(trace.trace_id) == trace


def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = str(tmp_path / "obs.sqlite3")
    ObservabilityStore(path).save(_trace("t2"))

    reopened = ObservabilityStore(path)

    assert reopened.get("t2") is not None
