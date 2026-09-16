"""SQLite-backed store for traces (memora.config.settings.observability_db_path).
Plain SQLite to start; revisit DuckDB if trace analytics need columnar speed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    retrieved TEXT NOT NULL,
    reranked TEXT NOT NULL,
    conflicts TEXT NOT NULL,
    context TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Trace:
    trace_id: str
    query: str
    retrieved: list[dict]
    reranked: list[dict]
    conflicts: list[dict]
    context: str
    response: str
    created_at: str


def _row_to_trace(row: tuple) -> Trace:
    return Trace(
        trace_id=row[0],
        query=row[1],
        retrieved=json.loads(row[2]),
        reranked=json.loads(row[3]),
        conflicts=json.loads(row[4]),
        context=row[5],
        response=row[6],
        created_at=row[7],
    )


class ObservabilityStore:
    def __init__(self, path: str) -> None:
        self._path = path
        conn = sqlite3.connect(self._path)
        try:
            conn.execute(_TABLE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def save(self, trace: Trace) -> None:
        conn = sqlite3.connect(self._path)
        try:
            conn.execute(
                "INSERT INTO traces "
                "(trace_id, query, retrieved, reranked, conflicts, context, response, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.query,
                    json.dumps(trace.retrieved),
                    json.dumps(trace.reranked),
                    json.dumps(trace.conflicts),
                    trace.context,
                    trace.response,
                    trace.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, trace_id: str) -> Trace | None:
        conn = sqlite3.connect(self._path)
        try:
            row = conn.execute(
                "SELECT trace_id, query, retrieved, reranked, conflicts, context, response, created_at "
                "FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        finally:
            conn.close()

        return None if row is None else _row_to_trace(row)

    def list_traces(self) -> list[Trace]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                "SELECT trace_id, query, retrieved, reranked, conflicts, context, response, created_at "
                "FROM traces ORDER BY created_at DESC"
            ).fetchall()
        finally:
            conn.close()

        return [_row_to_trace(row) for row in rows]
