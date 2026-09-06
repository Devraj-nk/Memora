# SQLite-backed store for traces (memora.config.settings.observability_db_path).
# Plain SQLite to start; revisit DuckDB if trace analytics need columnar speed.


class ObservabilityStore:
    def __init__(self, path: str) -> None:
        raise NotImplementedError
