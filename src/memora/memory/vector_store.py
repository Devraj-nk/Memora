"""LanceDB wrapper (memora.config.settings.vector_store_path) — embedded,
no separate DB server required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

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


def _modified_at_to_str(modified_at: datetime | None) -> str:
    # Stored as ISO text rather than a native LanceDB timestamp column so a
    # None (chunks built by hand, mostly in tests) never needs special-casing
    # against Arrow's schema inference - same convention observability.store
    # already uses for created_at.
    return modified_at.isoformat() if modified_at is not None else ""


def _modified_at_from_str(value: str) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


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
                "modified_at": _modified_at_to_str(c.modified_at),
                "vector": vector,
            }
            for c, vector in zip(chunks, embeddings)
        ]

        try:
            self._db.open_table(_TABLE_NAME).add(rows)
        except ValueError:
            self._db.create_table(_TABLE_NAME, data=rows)

    def all_chunks(self) -> list[Chunk]:
        return [chunk for chunk, _ in self.all_chunks_with_vectors()]

    def all_chunks_with_vectors(self) -> list[tuple[Chunk, list[float]]]:
        try:
            table = self._db.open_table(_TABLE_NAME)
        except ValueError:
            return []

        return [
            (
                Chunk(
                    text=r["text"],
                    source=r["source"],
                    chunk_index=r["chunk_index"],
                    start_offset=r["start_offset"],
                    end_offset=r["end_offset"],
                    modified_at=_modified_at_from_str(r["modified_at"]),
                ),
                r["vector"],
            )
            for r in table.to_arrow().to_pylist()
        ]

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
