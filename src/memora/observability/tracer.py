"""Records one retrieval/response cycle (query -> retrieved -> scores ->
reranking -> conflicts -> selected -> context -> response) as a trace.

Conflicts aren't recorded yet (knowledge_graph/ is still a stub, and
contradiction detection needs it) - a trace currently covers retrieval,
reranking, context, and response.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from memora.observability.store import ObservabilityStore, Trace
from memora.retrieval.hybrid import RetrievedChunk
from memora.retrieval.reranker import RerankedChunk


def record_trace(
    query: str,
    retrieved: list[RetrievedChunk],
    reranked: list[RerankedChunk],
    context: str,
    response: str,
    store: ObservabilityStore,
) -> str:
    trace = Trace(
        trace_id=str(uuid.uuid4()),
        query=query,
        retrieved=[asdict(chunk) for chunk in retrieved],
        reranked=[asdict(chunk) for chunk in reranked],
        context=context,
        response=response,
        created_at=datetime.now(UTC).isoformat(),
    )
    store.save(trace)
    return trace.trace_id
