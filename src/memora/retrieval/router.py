"""Orchestrates a query end to end: hybrid retrieval -> rerank -> context
builder -> LLM, emitting an observability trace at each stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from memora.knowledge_graph.graph_store import GraphStore
from memora.llm.client import generate
from memora.memory.context_builder import build_context
from memora.memory.vector_store import VectorStore
from memora.observability.store import ObservabilityStore
from memora.observability.tracer import record_trace
from memora.retrieval.hybrid import retrieve
from memora.retrieval.reranker import RerankedChunk, rerank

# Cast a wider net at the retrieval stage than we keep after reranking - the
# cross-encoder is more accurate but too slow to run over the whole corpus.
DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 5
DEFAULT_TOKEN_BUDGET = 2000

_PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the context doesn't contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RerankedChunk]
    trace_id: str


def route(
    query: str,
    store: VectorStore,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
    graph_store: GraphStore | None = None,
) -> list[RerankedChunk]:
    candidates = retrieve(query, store, top_k=candidate_k, graph_store=graph_store)
    return rerank(query, candidates)[:top_k]


def answer(
    query: str,
    store: VectorStore,
    observability_store: ObservabilityStore,
    graph_store: GraphStore | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> AnswerResult:
    retrieved = retrieve(query, store, top_k=candidate_k, graph_store=graph_store)
    reranked = rerank(query, retrieved)[:top_k]
    context = build_context(reranked, token_budget)
    response = generate(_PROMPT_TEMPLATE.format(context=context, query=query))

    conflicts = (
        graph_store.conflicts(set(graph_store.match_entities(query))) if graph_store is not None else None
    )
    trace_id = record_trace(query, retrieved, reranked, context, response, observability_store, conflicts)

    return AnswerResult(answer=response, sources=reranked, trace_id=trace_id)
