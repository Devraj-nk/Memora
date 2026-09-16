from fastapi import APIRouter, Depends
from pydantic import BaseModel

from memora.api.dependencies import get_graph_store, get_observability_store, get_vector_store
from memora.knowledge_graph.graph_store import GraphStore
from memora.memory.vector_store import VectorStore
from memora.observability.store import ObservabilityStore
from memora.retrieval.router import answer

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class RankedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[RankedChunk]
    trace_id: str


@router.post("/", response_model=QueryResponse)
def query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    observability_store: ObservabilityStore = Depends(get_observability_store),
    graph_store: GraphStore = Depends(get_graph_store),
) -> QueryResponse:
    result = answer(request.query, store, observability_store, graph_store)
    return QueryResponse(
        answer=result.answer,
        sources=[
            RankedChunk(text=s.text, source=s.source, chunk_index=s.chunk_index, score=s.score)
            for s in result.sources
        ],
        trace_id=result.trace_id,
    )
