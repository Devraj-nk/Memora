"""networkx graph persisted to SQLite (memora.config.settings.graph_db_path).
Swap for the optional `graph-db` (Neo4j) extra if query complexity outgrows this.

Entities/relationships come from ingestion.pipeline via
knowledge_graph.extractor, one (subject, relation, object) triple at a time,
each tagged with the (source, chunk_index) of the chunk it came from - that
provenance is what lets the graph act as a retrieval signal (which chunks
touch entities relevant to a query) and a conflict source (which subjects
have contradictory relations recorded).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import networkx as nx

from memora.knowledge_graph.extractor import Relationship

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS relationships (
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL
)
"""

DEFAULT_HOPS = 1


@dataclass(frozen=True)
class Conflict:
    subject: str
    relation: str
    objects: list[str]


class GraphStore:
    def __init__(self, path: str) -> None:
        self._path = path
        conn = sqlite3.connect(path)
        try:
            conn.execute(_TABLE_SCHEMA)
            conn.commit()
            rows = conn.execute("SELECT subject, relation, object, source, chunk_index FROM relationships").fetchall()
        finally:
            conn.close()

        self._graph = nx.MultiDiGraph()
        for subject, relation, obj, source, chunk_index in rows:
            self._graph.add_edge(subject, obj, relation=relation, source=source, chunk_index=chunk_index)

    def add(self, relationships: list[Relationship], source: str, chunk_index: int) -> None:
        if not relationships:
            return

        conn = sqlite3.connect(self._path)
        try:
            conn.executemany(
                "INSERT INTO relationships (subject, relation, object, source, chunk_index) VALUES (?, ?, ?, ?, ?)",
                [(r.subject, r.relation, r.object, source, chunk_index) for r in relationships],
            )
            conn.commit()
        finally:
            conn.close()

        for r in relationships:
            self._graph.add_edge(r.subject, r.object, relation=r.relation, source=source, chunk_index=chunk_index)

    def match_entities(self, text: str) -> list[str]:
        """Graph nodes that appear as a whole word/phrase in `text` (case-insensitive).

        A simple containment heuristic in lieu of running NER on every
        query - works for entities named literally in the query, which
        covers the common case for a personal knowledge base.
        """
        text_lower = text.lower()
        return [node for node in self._graph.nodes if re.search(rf"\b{re.escape(node.lower())}\b", text_lower)]

    def related_chunks(self, query: str, hops: int = DEFAULT_HOPS) -> list[tuple[str, int]]:
        """(source, chunk_index) pairs reachable within `hops` graph steps of
        any entity matched in `query`, ranked by how many matched edges touch
        each chunk (more connections = more likely relevant).
        """
        frontier = set(self.match_entities(query))
        if not frontier:
            return []

        seen_nodes = set(frontier)
        touched_edges: list[dict] = []

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, target, data in self._graph.out_edges(node, data=True):
                    touched_edges.append(data)
                    if target not in seen_nodes:
                        next_frontier.add(target)
                        seen_nodes.add(target)
                for source_node, _, data in self._graph.in_edges(node, data=True):
                    touched_edges.append(data)
                    if source_node not in seen_nodes:
                        next_frontier.add(source_node)
                        seen_nodes.add(source_node)
            frontier = next_frontier

        counts: dict[tuple[str, int], int] = {}
        for data in touched_edges:
            key = (data["source"], data["chunk_index"])
            counts[key] = counts.get(key, 0) + 1

        return sorted(counts, key=lambda k: counts[k], reverse=True)

    def conflicts(self, subjects: set[str] | None = None) -> list[Conflict]:
        """Subject+relation pairs recorded with more than one distinct object.

        A coarse heuristic, not true contradiction detection: "Memora | uses
        | LanceDB" and "Memora | uses | Neo4j" both being true would still
        flag here, since telling that apart from an actual contradiction
        ("Alice | lives in | Paris" vs "Alice | lives in | Tokyo") needs
        semantic understanding of the relation, not just string identity.
        Pass `subjects` to scope this to entities relevant to one query
        instead of the whole graph.
        """
        grouped: dict[tuple[str, str], set[str]] = {}
        for u, v, data in self._graph.edges(data=True):
            if subjects is not None and u not in subjects:
                continue
            key = (u, data["relation"])
            grouped.setdefault(key, set()).add(v)

        return [
            Conflict(subject=subject, relation=relation, objects=sorted(objects))
            for (subject, relation), objects in grouped.items()
            if len(objects) > 1
        ]
