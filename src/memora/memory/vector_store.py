"""LanceDB wrapper (memora.config.settings.vector_store_path) — embedded,
no separate DB server required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import lancedb

from memora.ingestion.chunker import Chunk

_TABLE_NAME = "chunks"


@dataclass(frozen=True)
class Match:
    text: str
    source: str
    chunk_index: int
    start_offset: int
    end_offset: int
    # L2 distance between query and chunk vectors — lower is more similar.
    # Vectors are expected pre-normalized (see ingestion.embedder), so this
    # ranks the same as cosine similarity would.
    distance: float


class VectorStore:
    def __init__(self, path: str) -> None:
        self._db = lancedb.connect(path)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        rows = [
            {
                "id": str(uuid.uuid4()),
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "vector": vector,
            }
            for c, vector in zip(chunks, embeddings)
        ]

        try:
            self._db.open_table(_TABLE_NAME).add(rows)
        except ValueError:
            self._db.create_table(_TABLE_NAME, data=rows)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[Match]:
        try:
            table = self._db.open_table(_TABLE_NAME)
        except ValueError:
            return []

        results = table.search(query_vector).limit(top_k).to_list()
        return [
            Match(
                text=r["text"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                start_offset=r["start_offset"],
                end_offset=r["end_offset"],
                distance=r["_distance"],
            )
            for r in results
        ]
