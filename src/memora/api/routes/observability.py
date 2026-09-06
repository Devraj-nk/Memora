from fastapi import APIRouter

router = APIRouter()

# GET /observability/traces/{trace_id} — full retrieval/rerank/response trace
# GET /observability/metrics — aggregate memory-quality metrics
