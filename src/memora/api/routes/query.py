from fastapi import APIRouter

router = APIRouter()

# POST /query — run the memory router: hybrid retrieval -> rerank -> context
# builder -> LLM, returning an answer plus sources and a trace id.
