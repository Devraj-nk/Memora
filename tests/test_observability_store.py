from pathlib import Path

from memora.observability.store import ObservabilityStore, Trace


def _trace(trace_id: str = "t1", created_at: str = "2026-09-13T00:00:00+00:00") -> Trace:
    return Trace(
        trace_id=trace_id,
        query="what is memora?",
        retrieved=[{"text": "a"}],
        reranked=[{"text": "a", "score": 0.9}],
        conflicts=[],
        context="[s.txt]\na",
        response="Memora is a memory engine.",
        created_at=created_at,
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


def test_list_traces_on_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))

    assert store.list_traces() == []


def test_list_traces_returns_all_saved_traces_newest_first(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    store.save(_trace("t1", created_at="2026-09-01T00:00:00+00:00"))
    store.save(_trace("t2", created_at="2026-09-02T00:00:00+00:00"))

    traces = store.list_traces()

    assert [t.trace_id for t in traces] == ["t2", "t1"]
