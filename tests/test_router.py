from pathlib import Path

import pytest

import memora.retrieval.router as router_module
from memora.ingestion.chunker import Chunk
from memora.ingestion.embedder import embed
from memora.memory.vector_store import VectorStore
from memora.observability.store import ObservabilityStore
from memora.retrieval.router import answer, route


def _add(store: VectorStore, texts: list[str]) -> None:
    chunks = [
        Chunk(text=t, source="s.txt", chunk_index=i, start_offset=0, end_offset=len(t))
        for i, t in enumerate(texts)
    ]
    store.add(chunks, embed(texts))


def test_route_on_empty_store_returns_no_results(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    assert route("anything", store) == []


def test_route_returns_most_relevant_chunk_first(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(
        store,
        [
            "The weather today is sunny with a light breeze.",
            "Python is a popular programming language for data science.",
            "Cats and dogs are common household pets.",
        ],
    )

    results = route("what programming language is popular for data science?", store)

    assert results[0].text == "Python is a popular programming language for data science."


def test_route_respects_top_k(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(store, [f"document number {i} about various topics" for i in range(5)])

    results = route("document topics", store, candidate_k=5, top_k=2)

    assert len(results) == 2


def test_answer_returns_generated_text_sources_and_a_trace_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VectorStore(str(tmp_path / "db"))
    obs_store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    _add(store, ["Python is a popular programming language for data science."])
    monkeypatch.setattr(router_module, "generate", lambda prompt: "Python.")

    result = answer("what language is popular for data science?", store, obs_store)

    assert result.answer == "Python."
    assert len(result.sources) == 1
    assert result.trace_id

    saved = obs_store.get(result.trace_id)
    assert saved is not None
    assert saved.response == "Python."


def test_answer_passes_query_and_context_to_the_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = VectorStore(str(tmp_path / "db"))
    obs_store = ObservabilityStore(str(tmp_path / "obs.sqlite3"))
    _add(store, ["Memora tracks how memory is retrieved and used."])

    captured_prompts: list[str] = []
    monkeypatch.setattr(
        router_module, "generate", lambda prompt: captured_prompts.append(prompt) or "ok"
    )

    answer("what does memora track?", store, obs_store)

    assert len(captured_prompts) == 1
    assert "what does memora track?" in captured_prompts[0]
    assert "Memora tracks how memory is retrieved and used." in captured_prompts[0]
