"""Cross-encoder reranking of hybrid.retrieve()'s candidates
(memora.config.settings.reranker_model).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from memora.config import settings
from memora.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    text: str
    source: str
    chunk_index: int
    start_offset: int
    end_offset: int
    # Cross-encoder relevance score for (query, text) - not comparable to
    # RetrievedChunk.score, which is a reciprocal-rank-fusion score.
    score: float


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str, candidates: list[RetrievedChunk], model_name: str | None = None
) -> list[RerankedChunk]:
    if not candidates:
        return []

    model = _get_model(model_name or settings.reranker_model)
    scores = model.predict([(query, c.text) for c in candidates])

    reranked = [
        RerankedChunk(
            text=c.text,
            source=c.source,
            chunk_index=c.chunk_index,
            start_offset=c.start_offset,
            end_offset=c.end_offset,
            score=float(score),
        )
        for c, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked
