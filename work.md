# Work Log

## 2026-09-06 — Tech stack decision + project scaffold

**Stack decided:** Python 3.11+, src-layout package `memora`.

- API: FastAPI + uvicorn
- Embeddings: `sentence-transformers`, model `sentence-transformers/all-MiniLM-L6-v2` (small/fast local default)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Vector store: LanceDB (embedded, no server process)
- Knowledge graph: `networkx` in-process, persisted to SQLite
- Keyword retrieval signal: `rank-bm25`
- LLM/SLM: **Ollama**, default model **`llama3.2:3b`** — local-first, runs comfortably on a personal machine; swap to a larger model (e.g. `qwen2.5:7b-instruct`) via `.env` (`LLM_MODEL`) if quality needs outweigh speed
- Observability store: plain SQLite (revisit DuckDB later if trace analytics need columnar speed)
- Config: `pydantic-settings`, `.env` file (see `.env.example`)

Config values live in [config.py](src/memora/config.py); defaults mirror [.env.example](.env.example).

**Project structure created** under `src/memora/`, mirroring the HLD in the README:

- `api/` — FastAPI app + routes (`ingest`, `query`, `observability`)
- `ingestion/` — parser, chunker, embedder
- `retrieval/` — hybrid retrieval, reranker, query router
- `knowledge_graph/` — entity/relationship extractor, graph store
- `memory/` — vector store, context builder
- `llm/` — Ollama client wrapper
- `observability/` — tracer, trace store, metrics

All modules are stubs (`raise NotImplementedError`) — structure only, no logic yet.

**External dependencies not covered by `pip install`:**

- **Ollama** must be installed separately (it's a standalone service, not a PyPI package). After installing, pull the default model: `ollama pull llama3.2:3b`. The app talks to it over HTTP at `OLLAMA_HOST` (default `http://localhost:11434`).
- **Tesseract OCR** + **Poppler** — only needed once ingestion handles scanned PDFs/images via the optional `parsing` extra (`unstructured[all-docs]`, `pytesseract`). Not required for plain text/markdown/code ingestion.
- **Neo4j** — optional, only if the knowledge graph outgrows in-process `networkx` + SQLite. Install via the `graph-db` extra (`pip install -e ".[graph-db]"`) plus a running Neo4j instance.

No other external DBs required — LanceDB and SQLite are both embedded (no server to run/manage).

**Next steps:** implement `ingestion/parser.py` for plain text/markdown first, wire up `memory/vector_store.py`, then get a trivial ingest → query round trip working end to end before adding the knowledge graph and reranking.

## 2026-09-07 — Parser implemented

Implemented [`ingestion/parser.py`](src/memora/ingestion/parser.py): `parse(source_path)` reads a file and returns a `ParsedDocument(text, source, mime_type, modified_at)`.

- Handles plain text/markdown/code by extension allowlist (`.txt`, `.md`, `.py`, `.json`, etc.) — reads as UTF-8, replacing undecodable bytes rather than raising.
- Raises `FileNotFoundError` for a missing path, `ValueError` for an unsupported extension (e.g. `.pdf`, `.docx`) with a pointer to the optional `parsing` extra.
- `mime_type` uses stdlib `mimetypes`, with explicit overrides for `.md`/`.markdown` since the OS-provided guess is inconsistent across platforms (notably Windows).
- Tests added in [`tests/test_parser.py`](tests/test_parser.py) covering a normal read, missing file, and unsupported extension.

**Next steps:** implement `ingestion/chunker.py` (split parsed text into retrieval-sized chunks), then `ingestion/embedder.py` and `memory/vector_store.py` to complete the ingest half of the round trip.
