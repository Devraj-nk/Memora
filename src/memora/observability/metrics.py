"""Aggregate memory-quality metrics computed from stored traces.

Only metrics derivable from what a trace actually records are implemented
here: retrieval volume, context/token cost, and a contradiction rate from
knowledge-graph conflicts. The README's other targets - retrieval
relevance/recall (need labeled ground truth to score against), duplicate
memories, memory freshness, and answer grounding (needs an entailment/NLI
check of the response against its context) - aren't computable from trace
data alone and are deliberately left out rather than faked.
"""

from __future__ import annotations

from dataclasses import dataclass

from memora.memory.context_builder import estimate_tokens
from memora.observability.store import ObservabilityStore


@dataclass(frozen=True)
class Metrics:
    trace_count: int
    avg_sources_per_trace: float
    avg_context_tokens: float
    contradiction_rate: float
    unique_sources_cited: int


def compute_metrics(store: ObservabilityStore) -> Metrics:
    traces = store.list_traces()
    if not traces:
        return Metrics(
            trace_count=0,
            avg_sources_per_trace=0.0,
            avg_context_tokens=0.0,
            contradiction_rate=0.0,
            unique_sources_cited=0,
        )

    n = len(traces)
    total_sources = sum(len(t.reranked) for t in traces)
    total_context_tokens = sum(estimate_tokens(t.context) for t in traces)
    contradictory = sum(1 for t in traces if t.conflicts)
    unique_sources = {chunk["source"] for t in traces for chunk in t.reranked}

    return Metrics(
        trace_count=n,
        avg_sources_per_trace=total_sources / n,
        avg_context_tokens=total_context_tokens / n,
        contradiction_rate=contradictory / n,
        unique_sources_cited=len(unique_sources),
    )
