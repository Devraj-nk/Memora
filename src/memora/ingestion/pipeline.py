"""End-to-end ingest: parser -> chunker -> embedder -> vector store
(+ knowledge graph extraction, when a GraphStore is given).
"""

from __future__ import annotations

from dataclasses import dataclass

from memora.ingestion.chunker import chunk
from memora.ingestion.embedder import embed
from memora.ingestion.parser import parse
from memora.knowledge_graph.extractor import extract
from memora.knowledge_graph.graph_store import GraphStore
from memora.memory.vector_store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunk_count: int


def ingest_source(source_path: str, store: VectorStore, graph_store: GraphStore | None = None) -> IngestResult:
    document = parse(source_path)
    chunks = chunk(document)

    if chunks:
        embeddings = embed([c.text for c in chunks])
        store.add(chunks, embeddings)

        if graph_store is not None:
            for c in chunks:
                relationships = extract(c.text)
                graph_store.add(relationships, c.source, c.chunk_index)

    return IngestResult(source=document.source, chunk_count=len(chunks))
