from memora.memory.context_builder import build_context
from memora.retrieval.reranker import RerankedChunk


def _chunk(text: str, source: str = "s.txt", chunk_index: int = 0) -> RerankedChunk:
    return RerankedChunk(
        text=text, source=source, chunk_index=chunk_index, start_offset=0, end_offset=len(text), score=1.0
    )


def test_build_context_empty_memories_returns_empty_string() -> None:
    assert build_context([], token_budget=1000) == ""


def test_build_context_includes_chunk_text_and_source() -> None:
    context = build_context([_chunk("hello world", source="notes.md")], token_budget=1000)

    assert "hello world" in context
    assert "notes.md" in context


def test_build_context_skips_chunks_that_would_exceed_budget() -> None:
    small = _chunk("short", chunk_index=0)
    huge = _chunk("a very long chunk of text " * 100, chunk_index=1)

    context = build_context([small, huge], token_budget=10)

    assert "short" in context
    assert "a very long chunk" not in context


def test_build_context_zero_budget_returns_empty_string() -> None:
    assert build_context([_chunk("anything")], token_budget=0) == ""
