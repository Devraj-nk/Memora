import pytest

import memora.knowledge_graph.extractor as extractor_module
from memora.knowledge_graph.extractor import Relationship, extract


def test_extract_parses_pipe_delimited_triples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extractor_module, "generate", lambda prompt: "Memora | uses | LanceDB\nMemora | uses | Ollama"
    )

    relationships = extract("Memora uses LanceDB and Ollama.")

    assert relationships == [
        Relationship(subject="Memora", relation="uses", object="LanceDB"),
        Relationship(subject="Memora", relation="uses", object="Ollama"),
    ]


def test_extract_skips_malformed_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extractor_module,
        "generate",
        lambda prompt: "not a triple\nMemora | uses | LanceDB\n| missing | subject\nMemora | | empty relation",
    )

    relationships = extract("some text")

    assert relationships == [Relationship(subject="Memora", relation="uses", object="LanceDB")]


def test_extract_returns_empty_list_for_blank_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extractor_module, "generate", lambda prompt: "")

    assert extract("some text") == []


def test_extract_passes_the_text_into_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(extractor_module, "generate", lambda prompt: captured.append(prompt) or "")

    extract("Memora is local-first.")

    assert "Memora is local-first." in captured[0]
