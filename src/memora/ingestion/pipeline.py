"""End-to-end ingest: parser -> chunker -> embedder -> vector store."""

from __future__ import annotations

from dataclasses import dataclass

from memora.ingestion.chunker import chunk
from memora.ingestion.embedder import embed
from memora.ingestion.parser import parse
from memora.memory.vector_store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunk_count: int


def ingest_source(source_path: str, store: VectorStore) -> IngestResult:
    document = parse(source_path)
    chunks = chunk(document)

    if chunks:
        embeddings = embed([c.text for c in chunks])
        store.add(chunks, embeddings)

    return IngestResult(source=document.source, chunk_count=len(chunks))
