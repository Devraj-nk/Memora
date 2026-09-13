from pathlib import Path

from memora.observability.store import ObservabilityStore
from memora.observability.tracer import record_trace
from memora.retrieval.hybrid import RetrievedChunk
from memora.retrieval.reranker import RerankedChunk


def test_record_trace_persists_and_returns_a_trace_id(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    retrieved = [RetrievedChunk(text="a", source="s.txt", chunk_index=0, start_offset=0, end_offset=1, score=0.5)]
    reranked = [RerankedChunk(text="a", source="s.txt", chunk_index=0, start_offset=0, end_offset=1, score=2.1)]

    trace_id = record_trace("query", retrieved, reranked, "[s.txt]\na", "the answer", store)

    saved = store.get(trace_id)
    assert saved is not None
    assert saved.query == "query"
    assert saved.response == "the answer"
    assert saved.retrieved[0]["text"] == "a"
    assert saved.reranked[0]["score"] == 2.1


def test_record_trace_generates_unique_ids(tmp_path: Path) -> None:
    store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))

    id_a = record_trace("q1", [], [], "", "a", store)
    id_b = record_trace("q2", [], [], "", "b", store)

    assert id_a != id_b
