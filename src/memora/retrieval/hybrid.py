"""Combines keyword (rank-bm25) and semantic (vector store) signals into a
single ranked candidate set for the reranker.

Knowledge-graph signals are not wired in yet (knowledge_graph/ is still a
stub) - candidates come from these two signals only for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rank_bm25 import BM25Okapi

from memora.ingestion.embedder import embed
from memora.memory.vector_store import VectorStore


class _HasChunkFields(Protocol):
    text: str
    source: str
    chunk_index: int
    start_offset: int
    end_offset: int


DEFAULT_TOP_K = 10
# Reciprocal rank fusion constant (Cormack et al., 2009) - large enough that
# a single signal's #1 pick doesn't automatically dominate the other
# signal's #1 pick once both are combined.
_RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    start_offset: int
    end_offset: int
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def retrieve(query: str, store: VectorStore, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    chunks = store.all_chunks()
    if not chunks:
        return []

    # Semantic ranking: nearest chunks to the query embedding.
    query_vector = embed([query])[0]
    vector_ranked = store.search(query_vector, top_k=len(chunks))

    # Keyword ranking: BM25 over the full corpus.
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    keyword_scores = bm25.get_scores(_tokenize(query))
    keyword_ranked = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: keyword_scores[i], reverse=True)]

    # Reciprocal rank fusion: each signal votes by rank rather than raw
    # score, so BM25 scores and L2 distances (different scales entirely)
    # never need to be normalized against each other.
    fused_scores: dict[tuple[str, int], float] = {}
    fused_chunks: dict[tuple[str, int], _HasChunkFields] = {}

    for ranked_list in (vector_ranked, keyword_ranked):
        for rank, item in enumerate(ranked_list):
            key = (item.source, item.chunk_index)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            fused_chunks.setdefault(key, item)

    ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            text=fused_chunks[key].text,
            source=fused_chunks[key].source,
            chunk_index=fused_chunks[key].chunk_index,
            start_offset=fused_chunks[key].start_offset,
            end_offset=fused_chunks[key].end_offset,
            score=fused_scores[key],
        )
        for key in ranked_keys
    ]
