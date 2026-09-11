from pathlib import Path

from memora.ingestion.chunker import Chunk
from memora.ingestion.embedder import embed
from memora.memory.vector_store import VectorStore
from memora.retrieval.hybrid import retrieve


def _add(store: VectorStore, texts: list[str]) -> None:
    chunks = [
        Chunk(text=t, source="s.txt", chunk_index=i, start_offset=0, end_offset=len(t))
        for i, t in enumerate(texts)
    ]
    store.add(chunks, embed(texts))


def test_retrieve_on_empty_store_returns_no_matches(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))

    assert retrieve("anything", store) == []


def test_retrieve_finds_exact_keyword_match_even_if_semantically_unrelated(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(
        store,
        [
            "The quarterly earnings report showed strong revenue growth.",
            "Zephyrblorp is a rare artifact mentioned only once in the archive.",
            "Cats and dogs are common household pets.",
        ],
    )

    results = retrieve("zephyrblorp", store, top_k=3)

    assert results[0].text == "Zephyrblorp is a rare artifact mentioned only once in the archive."


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(store, [f"document number {i} about various topics" for i in range(5)])

    results = retrieve("document topics", store, top_k=2)

    assert len(results) == 2


def test_retrieve_results_are_sorted_by_score_descending(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "db"))
    _add(
        store,
        [
            "Python is a popular programming language for data science.",
            "The weather today is sunny with a light breeze.",
        ],
    )

    results = retrieve("python programming", store, top_k=2)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
