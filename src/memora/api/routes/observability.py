from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memora.api.dependencies import get_observability_store, get_vector_store
from memora.memory.vector_store import VectorStore
from memora.observability.evaluation import DEFAULT_SAMPLE_SIZE, DEFAULT_TOP_K, evaluate_retrieval
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
    answer_grounding_rate: float
    avg_memory_age_days: float | None
    duplicate_chunk_pairs: int


class EvaluateRequest(BaseModel):
    top_k: int = DEFAULT_TOP_K
    sample_size: int = DEFAULT_SAMPLE_SIZE


class EvaluateResponse(BaseModel):
    recall_at_k: float
    sample_size: int
    top_k: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str, store: ObservabilityStore = Depends(get_observability_store)) -> TraceResponse:
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    return TraceResponse(**asdict(trace))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    store: ObservabilityStore = Depends(get_observability_store),
    vector_store: VectorStore = Depends(get_vector_store),
) -> MetricsResponse:
    return MetricsResponse(**asdict(compute_metrics(store, vector_store)))


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    request: EvaluateRequest, vector_store: VectorStore = Depends(get_vector_store)
) -> EvaluateResponse:
    # Runs one LLM call per sampled chunk (synthetic question generation) -
    # noticeably slower than every other route here, by design: this is a
    # deliberately-triggered eval job, not something to run on every
    # dashboard load. See observability.evaluation's module docstring.
    result = evaluate_retrieval(vector_store, top_k=request.top_k, sample_size=request.sample_size)
    return EvaluateResponse(recall_at_k=result.recall_at_k, sample_size=result.sample_size, top_k=result.top_k)
