from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    vector_store_path: str = "data/db/vectors.lance"
    graph_db_path: str = "data/db/graph.sqlite3"
    observability_db_path: str = "data/db/observability.sqlite3"
    raw_data_path: str = "data/raw"


settings = Settings()
