"""Wraps sentence-transformers (memora.config.settings.embedding_model)
to embed chunks for the vector store, and queries at retrieval time.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from memora.config import settings


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []

    model = _get_model(model_name or settings.embedding_model)
    # Normalized so cosine similarity reduces to a dot product, which is
    # what the vector store (LanceDB) will search on.
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()
