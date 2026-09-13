"""Shared FastAPI dependencies - singletons injected via Depends() so tests
can swap them out with app.dependency_overrides instead of touching
data/db/ directly.
"""

from __future__ import annotations

from functools import lru_cache

from memora.config import settings
from memora.memory.vector_store import VectorStore
from memora.observability.store import ObservabilityStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings.vector_store_path)


@lru_cache(maxsize=1)
def get_observability_store() -> ObservabilityStore:
    return ObservabilityStore(settings.observability_db_path)
