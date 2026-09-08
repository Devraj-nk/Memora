"""Splits parsed text into retrieval-sized chunks, preserving enough
metadata (source, offset) to trace a chunk back to its origin.
"""

from __future__ import annotations

from dataclasses import dataclass

from memora.ingestion.parser import ParsedDocument

DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 150  # characters


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int
    start_offset: int
    end_offset: int


def chunk(
    document: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.text
    if not text.strip():
        return []

    text_len = len(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Avoid cutting mid-word: back off to the nearest whitespace in
        # range, unless this is the final chunk.
        if end < text_len:
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    source=document.source,
                    chunk_index=index,
                    start_offset=start,
                    end_offset=end,
                )
            )
            index += 1

        if end >= text_len:
            break

        # Step forward by chunk_size minus overlap; if the whitespace
        # backoff above left no room for that, advance to `end` instead so
        # we always make forward progress.
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks
