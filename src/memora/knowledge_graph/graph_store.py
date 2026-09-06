# networkx graph persisted to SQLite (memora.config.settings.graph_db_path).
# Swap for the optional `graph-db` (Neo4j) extra if query complexity outgrows this.


class GraphStore:
    def __init__(self, path: str) -> None:
        raise NotImplementedError
