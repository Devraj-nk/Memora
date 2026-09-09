from memora.ingestion.embedder import embed


def test_embed_returns_one_vector_per_text() -> None:
    vectors = embed(["hello world", "another sentence"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


def test_embed_empty_list_returns_empty_list() -> None:
    assert embed([]) == []


def test_embed_is_deterministic_for_same_text() -> None:
    a = embed(["deterministic check"])[0]
    b = embed(["deterministic check"])[0]

    assert a == b


def test_embed_similar_texts_are_closer_than_dissimilar() -> None:
    query, similar, different = embed(
        ["a dog playing in the park", "a puppy running in a field", "quarterly tax filing deadline"]
    )

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert dot(query, similar) > dot(query, different)
