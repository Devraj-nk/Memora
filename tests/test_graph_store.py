from pathlib import Path

from memora.knowledge_graph.extractor import Relationship
from memora.knowledge_graph.graph_store import GraphStore


def test_related_chunks_on_empty_graph_returns_nothing(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))

    assert store.related_chunks("anything") == []


def test_add_then_related_chunks_finds_chunk_touching_matched_entity(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Memora", relation="uses", object="LanceDB")], source="s.txt", chunk_index=0)

    assert store.related_chunks("What does Memora use?") == [("s.txt", 0)]


def test_related_chunks_ignores_unmatched_query(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Memora", relation="uses", object="LanceDB")], source="s.txt", chunk_index=0)

    assert store.related_chunks("something unrelated entirely") == []


def test_related_chunks_two_hops_reaches_indirect_chunk(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Memora", relation="uses", object="LanceDB")], source="a.txt", chunk_index=0)
    store.add(
        [Relationship(subject="LanceDB", relation="is a", object="vector database")],
        source="b.txt",
        chunk_index=1,
    )

    one_hop = set(store.related_chunks("Memora", hops=1))
    two_hop = set(store.related_chunks("Memora", hops=2))

    assert ("a.txt", 0) in one_hop
    assert ("b.txt", 1) not in one_hop
    assert ("b.txt", 1) in two_hop


def test_graph_persists_across_instances(tmp_path: Path) -> None:
    path = str(tmp_path / "graph.sqlite3")
    GraphStore(path).add([Relationship(subject="Memora", relation="uses", object="LanceDB")], "s.txt", 0)

    reopened = GraphStore(path)

    assert reopened.related_chunks("Memora") == [("s.txt", 0)]


def test_conflicts_flags_same_subject_relation_different_objects(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Alice", relation="lives in", object="Paris")], "a.txt", 0)
    store.add([Relationship(subject="Alice", relation="lives in", object="Tokyo")], "b.txt", 0)

    conflicts = store.conflicts()

    assert len(conflicts) == 1
    assert conflicts[0].subject == "Alice"
    assert conflicts[0].objects == ["Paris", "Tokyo"]


def test_conflicts_ignores_consistent_relationships(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Alice", relation="lives in", object="Paris")], "a.txt", 0)
    store.add([Relationship(subject="Alice", relation="lives in", object="Paris")], "b.txt", 1)

    assert store.conflicts() == []


def test_conflicts_can_be_scoped_to_subjects(tmp_path: Path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite3"))
    store.add([Relationship(subject="Alice", relation="lives in", object="Paris")], "a.txt", 0)
    store.add([Relationship(subject="Alice", relation="lives in", object="Tokyo")], "b.txt", 0)
    store.add([Relationship(subject="Bob", relation="lives in", object="Berlin")], "c.txt", 0)
    store.add([Relationship(subject="Bob", relation="lives in", object="Rome")], "d.txt", 0)

    conflicts = store.conflicts(subjects={"Alice"})

    assert len(conflicts) == 1
    assert conflicts[0].subject == "Alice"
