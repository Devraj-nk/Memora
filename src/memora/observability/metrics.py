"""Aggregate memory-quality metrics computed from stored traces and the
vector store.

Metrics fall into three groups:
- trace-derived: volume, context/token cost, contradiction rate, and answer
  grounding (an NLI check of each trace's stored context+response - see
  observability.grounding.is_grounded - run lazily here rather than at
  query time, so POST /query/ never pays for it).
- corpus-derived (need a VectorStore, not just the trace store): memory
  freshness and duplicate memories.
- deliberately NOT computed here: retrieval relevance/recall has its own
  module (observability.evaluation) since it needs an LLM call per sampled
  chunk and isn't cheap enough to run inline on every metrics request.
  Source quality is left undefined - the README doesn't give it a concrete
  rubric to implement against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from memora.memory.context_builder import estimate_tokens
from memora.memory.vector_store import VectorStore
from memora.observability.grounding import is_grounded
from memora.observability.store import ObservabilityStore

DUPLICATE_SIMILARITY_THRESHOLD = 0.95


@dataclass(frozen=True)
class Metrics:
    trace_count: int
    avg_sources_per_trace: float
    avg_context_tokens: float
    contradiction_rate: float
    unique_sources_cited: int
    answer_grounding_rate: float
    avg_memory_age_days: float | None
    duplicate_chunk_pairs: int


def compute_metrics(store: ObservabilityStore, vector_store: VectorStore) -> Metrics:
    traces = store.list_traces()

    if traces:
        n = len(traces)
        total_sources = sum(len(t.reranked) for t in traces)
        total_context_tokens = sum(estimate_tokens(t.context) for t in traces)
        contradictory = sum(1 for t in traces if t.conflicts)
        unique_sources = {chunk["source"] for t in traces for chunk in t.reranked}
        grounded = sum(1 for t in traces if is_grounded(t.context, t.query, t.response))

        avg_sources_per_trace = total_sources / n
        avg_context_tokens = total_context_tokens / n
        contradiction_rate = contradictory / n
        unique_sources_cited = len(unique_sources)
        answer_grounding_rate = grounded / n
    else:
        n = 0
        avg_sources_per_trace = 0.0
        avg_context_tokens = 0.0
        contradiction_rate = 0.0
        unique_sources_cited = 0
        answer_grounding_rate = 0.0

    return Metrics(
        trace_count=n,
        avg_sources_per_trace=avg_sources_per_trace,
        avg_context_tokens=avg_context_tokens,
        contradiction_rate=contradiction_rate,
        unique_sources_cited=unique_sources_cited,
        answer_grounding_rate=answer_grounding_rate,
        avg_memory_age_days=_compute_avg_memory_age_days(vector_store),
        duplicate_chunk_pairs=_count_duplicate_chunk_pairs(vector_store),
    )


def _compute_avg_memory_age_days(vector_store: VectorStore) -> float | None:
    ages = [
        (datetime.now(UTC) - chunk.modified_at).total_seconds() / 86400
        for chunk in vector_store.all_chunks()
        if chunk.modified_at is not None
    ]
    return sum(ages) / len(ages) if ages else None


def _count_duplicate_chunk_pairs(
    vector_store: VectorStore, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD
) -> int:
    chunks_with_vectors = vector_store.all_chunks_with_vectors()
    n = len(chunks_with_vectors)
    if n < 2:
        return 0

    # Vectors are pre-normalized (see ingestion.embedder), so cosine
    # similarity is just a dot product - no need to divide by norms.
    # O(n^2) similarity matrix: fine for a personal-scale corpus, would
    # need a proper nearest-neighbor index if this ever needs to scale.
    vectors = np.array([vector for _, vector in chunks_with_vectors])
    similarity = vectors @ vectors.T
    upper_triangle = np.triu(similarity, k=1)
    return int((upper_triangle >= threshold).sum())
