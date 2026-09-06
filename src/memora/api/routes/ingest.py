from fastapi import APIRouter

router = APIRouter()

# POST /ingest — accept a document/note/URL, hand off to the ingestion pipeline
# (parser -> chunker -> embedder -> vector store + knowledge graph).
