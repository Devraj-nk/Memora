from pathlib import Path

import pytest

import memora.observability.evaluation as evaluation_module
from memora.ingestion.chunker import Chunk
from memora.ingestion.embedder import embed
from memora.memory.vector_store import VectorStore
from memora.observability.evaluation import RetrievalEvalResult, evaluate_retrieval
from memora.retrieval.hybrid import RetrievedChunk


def _add(store: VectorStore, texts: list[str]) -> None:
    chunks = [
        Chunk(text=t, source="s.txt", chunk_index=i, start_offset=0, end_offset=len(t))
        for i, t in enumerate(texts)
    ]
    store.add(chunks, embed(texts))


def test_evaluate_retrieval_on_empty_store_returns_zero(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    result = evaluate_retrieval(store)

    assert result == RetrievalEvalResult(recall_at_k=0.0, sample_size=0, top_k=5)


def test_evaluate_retrieval_counts_hits_and_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolates evaluate_retrieval's own logic (sampling + hit counting) from
    # retrieve()'s actual ranking behavior, which test_hybrid.py already
    # covers - a fixed fake result makes "chunk 0" always a hit and
    # "chunk 1" always a miss, regardless of sampling order.
    store = VectorStore(str(tmp_path / "db"))
    _add(store, ["chunk one text", "chunk two text"])
    monkeypatch.setattr(evaluation_module, "generate_synthetic_question", lambda text: "a question")
    monkeypatch.setattr(
        evaluation_module,
        "retrieve",
        lambda query, store, top_k=5: [
            RetrievedChunk(text="chunk one text", source="s.txt", chunk_index=0, start_offset=0, end_offset=1, score=1.0)
        ],
    )

    result = evaluate_retrieval(store, top_k=1, sample_size=2, seed=0)

    assert result.sample_size == 2
    assert result.top_k == 1
    assert result.recall_at_k == 0.5


def test_evaluate_retrieval_respects_sample_size_smaller_than_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(store, [f"document number {i}" for i in range(5)])
    monkeypatch.setattr(evaluation_module, "generate_synthetic_question", lambda text: "a question")
    monkeypatch.setattr(evaluation_module, "retrieve", lambda query, store, top_k=5: [])

    result = evaluate_retrieval(store, sample_size=2, seed=0)

    assert result.sample_size == 2
    assert result.recall_at_k == 0.0


def test_evaluate_retrieval_passes_chunk_text_to_question_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(store, ["a specific piece of text"])
    captured: list[str] = []
    monkeypatch.setattr(
        evaluation_module, "generate_synthetic_question", lambda text: captured.append(text) or "q"
    )
    monkeypatch.setattr(evaluation_module, "retrieve", lambda query, store, top_k=5: [])

    evaluate_retrieval(store, sample_size=1)

    assert captured == ["a specific piece of text"]
