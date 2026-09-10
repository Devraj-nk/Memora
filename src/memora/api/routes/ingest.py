from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memora.config import settings
from memora.ingestion.pipeline import ingest_source
from memora.memory.vector_store import VectorStore

router = APIRouter()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings.vector_store_path)


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    source: str
    chunk_count: int


@router.post("/", response_model=IngestResponse)
def ingest(request: IngestRequest, store: VectorStore = Depends(get_vector_store)) -> IngestResponse:
    try:
        result = ingest_source(request.path, store)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Source not found: {request.path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return IngestResponse(source=result.source, chunk_count=result.chunk_count)
