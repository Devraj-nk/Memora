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

## 2026-09-08 — Chunker implemented

Implemented [`ingestion/chunker.py`](src/memora/ingestion/chunker.py): `chunk(document, chunk_size, chunk_overlap)` takes a `ParsedDocument` and returns a list of `Chunk(text, source, chunk_index, start_offset, end_offset)`.

- Character-based sliding window (default 1000 chars, 150 overlap) — simple and tokenizer-agnostic; can move to token-based sizing once `ingestion/embedder.py` picks a tokenizer.
- Backs off to the nearest whitespace boundary within range so chunks don't cut mid-word (skipped on the final chunk).
- Guards forward progress explicitly: if the whitespace backoff would make the next `start` stall or go backwards, it advances to `end` instead — avoids a subtle infinite-loop edge case with small `chunk_size` values.
- Raises `ValueError` if `chunk_overlap >= chunk_size`; returns `[]` for blank/whitespace-only documents.
- Tests added in [`tests/test_chunker.py`](tests/test_chunker.py): single-chunk short text, multi-chunk overlap on long text, empty text, and the overlap/size validation error. Full suite (`pytest tests/`) passes, 7/7.

**Next steps:** implement `ingestion/embedder.py` (sentence-transformers, `all-MiniLM-L6-v2`) and `memory/vector_store.py` (LanceDB) to complete the ingest half of the round trip.

## 2026-09-09 — Embedder implemented

Implemented [`ingestion/embedder.py`](src/memora/ingestion/embedder.py): `embed(texts, model_name=None)` returns one vector per input text.

- Loads `sentence-transformers` (default model from `settings.embedding_model`, i.e. `all-MiniLM-L6-v2`) lazily and caches it with `lru_cache` keyed on model name, so the model is loaded once per process rather than per call.
- Embeddings are L2-normalized (`normalize_embeddings=True`) so cosine similarity reduces to a dot product — matches how the upcoming LanceDB vector store will search.
- `embed([])` short-circuits to `[]` without touching the model.
- Tests added in [`tests/test_embedder.py`](tests/test_embedder.py): shape/dimension check, empty input, determinism, and a semantic sanity check (similar sentences score closer than unrelated ones on dot product).

**Next steps:** implement `memory/vector_store.py` (LanceDB) so ingested chunks + embeddings can actually be persisted and searched, then wire `ingestion/parser.py` → `chunker.py` → `embedder.py` → `vector_store.py` into a single ingest path (likely called from `api/routes/ingest.py`).

## 2026-09-10 — Vector store implemented

Implemented [`memory/vector_store.py`](src/memora/memory/vector_store.py): `VectorStore(path)` wraps LanceDB with `add(chunks, embeddings)` and `search(query_vector, top_k)` -> `list[Match]`.

- `add`/`search` open the `chunks` table lazily and catch the `ValueError` LanceDB raises for a missing table to decide create-vs-open — deliberately avoided `table_names()` (deprecated) and `list_tables()`/`table_exists()` (version-fragile: `table_exists` isn't even implemented for local connections on lancedb 0.38) in favor of this more stable open/create pattern. Verified directly against the installed lancedb 0.38.0 API before writing this.
- `Match.distance` is LanceDB's raw L2 distance (lower = more similar). Since `ingestion/embedder.py` already L2-normalizes vectors, ranking by L2 distance agrees with ranking by cosine similarity, so no extra normalization needed here.
- `add([], [])` is a no-op; mismatched `chunks`/`embeddings` lengths raise `ValueError`; `search` on an empty/nonexistent store returns `[]` instead of erroring.
- Tests in [`tests/test_vector_store.py`](tests/test_vector_store.py) use `tmp_path` for an isolated on-disk LanceDB per test: empty-store search, nearest-match ranking, accumulation across multiple `add()` calls, no-op empty add, and the length-mismatch error. Full suite: 16/16 passing.

This completes the ingest half of the pipeline end to end: `parser` → `chunker` → `embedder` → `vector_store`.

**Next steps:** wire these four into a single ingest function callable from `api/routes/ingest.py`, then start on `retrieval/hybrid.py` (needs this vector store's `search`, plus a keyword/BM25 signal) to begin the query half.
