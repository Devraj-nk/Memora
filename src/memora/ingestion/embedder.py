# Wraps sentence-transformers (memora.config.settings.embedding_model)
# to embed chunks for the vector store, and queries at retrieval time.


def embed(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError
