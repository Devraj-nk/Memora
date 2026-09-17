import pytest

from memora.observability.grounding import is_grounded


def test_is_grounded_empty_context_is_never_grounded() -> None:
    assert is_grounded("", "any query", "any answer at all") is False


@pytest.mark.slow
def test_is_grounded_true_when_context_supports_the_answer() -> None:
    context = "Memora uses LanceDB as its vector store."
    query = "What vector store does Memora use?"

    assert is_grounded(context, query, "LanceDB.") is True


@pytest.mark.slow
def test_is_grounded_false_when_context_contradicts_the_answer() -> None:
    context = "Memora uses LanceDB as its vector store."
    query = "What vector store does Memora use?"

    assert is_grounded(context, query, "Neo4j.") is False


@pytest.mark.slow
def test_is_grounded_strips_context_builder_source_header() -> None:
    # context_builder.build_context() prefixes blocks with "[source]\n" -
    # confirmed against the real model that leaving it in flips a correctly
    # grounded answer to "neutral" instead of "entailment".
    context = "[C:\\notes\\note.md]\nMemora uses LanceDB as its vector store."
    query = "What vector store does Memora use?"

    assert is_grounded(context, query, "LanceDB") is True
