from memora.retrieval.hybrid import RetrievedChunk
from memora.retrieval.reranker import rerank


def _candidate(text: str, chunk_index: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="s.txt",
        chunk_index=chunk_index,
        start_offset=0,
        end_offset=len(text),
        score=0.0,
    )


def test_rerank_empty_candidates_returns_empty_list() -> None:
    assert rerank("anything", []) == []


def test_rerank_orders_most_relevant_candidate_first() -> None:
    candidates = [
        _candidate("The weather today is sunny with a light breeze.", 0),
        _candidate("Python is a popular programming language for data science.", 1),
    ]

    results = rerank("what programming language is popular for data science?", candidates)

    assert results[0].text == "Python is a popular programming language for data science."
    assert results[0].score > results[1].score


def test_rerank_preserves_chunk_metadata() -> None:
    candidates = [_candidate("Memora tracks how memory is retrieved and used.", 3)]

    results = rerank("memory retrieval", candidates)

    assert results[0].source == "s.txt"
    assert results[0].chunk_index == 3
