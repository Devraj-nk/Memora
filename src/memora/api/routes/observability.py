from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memora.api.dependencies import get_observability_store
from memora.observability.metrics import compute_metrics
from memora.observability.store import ObservabilityStore

router = APIRouter()


class TraceResponse(BaseModel):
    trace_id: str
    query: str
    retrieved: list[dict]
    reranked: list[dict]
    conflicts: list[dict]
    context: str
    response: str
    created_at: str


class MetricsResponse(BaseModel):
    trace_count: int
    avg_sources_per_trace: float
    avg_context_tokens: float
    contradiction_rate: float
    unique_sources_cited: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str, store: ObservabilityStore = Depends(get_observability_store)) -> TraceResponse:
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    return TraceResponse(**asdict(trace))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(store: ObservabilityStore = Depends(get_observability_store)) -> MetricsResponse:
    return MetricsResponse(**asdict(compute_metrics(store)))
