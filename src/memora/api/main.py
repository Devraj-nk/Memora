from fastapi import FastAPI

from memora.api.routes import ingest, observability, query

app = FastAPI(title="Memora")

app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(observability.router, prefix="/observability", tags=["observability"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
