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

## 2026-09-11 — Ingest router implemented

Added [`ingestion/pipeline.py`](src/memora/ingestion/pipeline.py): `ingest_source(source_path, store)` runs `parse` → `chunk` → `embed` → `store.add`, returning `IngestResult(source, chunk_count)`. Skips embedding/storage entirely for a blank document (`chunk_count == 0`, nothing written).

Wired [`api/routes/ingest.py`](src/memora/api/routes/ingest.py): `POST /ingest/` takes `{"path": "..."}`, runs the pipeline, returns `{"source", "chunk_count"}`.

- `VectorStore` is a singleton via `get_vector_store()` (`@lru_cache`, path from `settings.vector_store_path`), injected with FastAPI's `Depends` rather than instantiated inline — lets tests swap in a `tmp_path`-backed store via `app.dependency_overrides` instead of touching `data/db/`.
- Maps pipeline errors to HTTP: `FileNotFoundError` → 404, `ValueError` (unsupported extension, from `parser.parse`) → 400.
- Confirmed with `TestClient` that `POST /ingest` (no trailing slash) 307-redirects to `/ingest/` and still reaches the handler — both forms work, tests hit `/ingest/` directly to skip the extra hop.
- Only accepts a local file `path` for now, matching what `ingestion/parser.py` actually supports — URL/raw-text ingestion (per the README's "web content" ambition) is future work, not faked here.

Tests: [`tests/test_pipeline.py`](tests/test_pipeline.py) (pipeline function directly, no HTTP) and [`tests/test_ingest_route.py`](tests/test_ingest_route.py) (through `TestClient`, covering success, missing file, unsupported extension). Full suite: 21/21 passing.

**Next steps:** start `retrieval/hybrid.py` (vector store `search` + a keyword/BM25 signal) and `retrieval/router.py` to begin the query half; a `GET/POST /query` route can follow the same `Depends`-singleton pattern used here.

## 2026-09-12 — Hybrid retrieval implemented

Implemented [`retrieval/hybrid.py`](src/memora/retrieval/hybrid.py): `retrieve(query, store, top_k=10)` combines the vector store's semantic `search` with a `rank-bm25` keyword signal into one ranked `list[RetrievedChunk]`.

- Both signals rank over the *full* corpus (`store.all_chunks()` for BM25, `store.search(top_k=len(chunks))` for semantic), then combine via reciprocal rank fusion (RRF, k=60) — fusing by rank rather than raw score sidesteps normalizing BM25 scores against L2 distances, which live on unrelated scales.
- Added `VectorStore.all_chunks()` ([memory/vector_store.py](src/memora/memory/vector_store.py)) to support this — reads the full table via `to_arrow().to_pylist()`, not `to_pandas()`, since `pandas` isn't an installed dependency here.
- Knowledge-graph signal is intentionally not wired in yet (`knowledge_graph/` is still a stub) — module docstring notes this so it isn't mistaken for an oversight.
- `retrieve` returns `[]` immediately on an empty store, skipping both signals.
- Tests: [`tests/test_hybrid.py`](tests/test_hybrid.py) (empty store, an exact-keyword match that BM25 surfaces even though it's semantically unrelated to the rest of the corpus, `top_k` limiting, descending score order) and two new cases in [`tests/test_vector_store.py`](tests/test_vector_store.py) for `all_chunks()`. Full suite: 27/27 passing.

**Next steps:** implement `retrieval/reranker.py` (cross-encoder over `retrieve()`'s candidates) and `retrieval/router.py` to tie parsing/routing of a query together, then wire a `GET/POST /query` route the way `ingest.py` wires the ingest pipeline.

## 2026-09-12 (cont.) — Reranker implemented

Implemented [`retrieval/reranker.py`](src/memora/retrieval/reranker.py): `rerank(query, candidates, model_name=None)` scores each `hybrid.RetrievedChunk` with a cross-encoder (`sentence-transformers/cross-encoder`, default model from `settings.reranker_model`) and returns `list[RerankedChunk]` sorted by relevance score, descending.

- Follows `ingestion/embedder.py`'s lazy-load-and-cache pattern (`lru_cache` on `_get_model`, keyed on model name) so the cross-encoder loads once per process.
- New `RerankedChunk` dataclass rather than reusing `RetrievedChunk` — its `score` is a cross-encoder relevance score, not comparable to `RetrievedChunk.score` (an RRF fusion score), so keeping them distinct types avoids silently mixing the two.
- `rerank(query, [])` short-circuits to `[]` without touching the model.
- Tests in [`tests/test_reranker.py`](tests/test_reranker.py): empty input, correct ordering on an obviously-more-relevant candidate, and metadata (source/chunk_index) preserved through reranking. Full suite: 30/30 passing.

**Next steps:** implement `retrieval/router.py` to tie `hybrid.retrieve()` → `reranker.rerank()` together end to end, then wire a `GET/POST /query` route (same `Depends`-singleton pattern as `ingest.py`).

## 2026-09-13 — Query router + `/query` route implemented

Implemented [`retrieval/router.py`](src/memora/retrieval/router.py): `route(query, store, candidate_k=20, top_k=5)` calls `hybrid.retrieve()` for a wide candidate pool, then `reranker.rerank()` to narrow it to the final `top_k`. Context building and LLM generation stay out of scope — `memory/context_builder.py` and `llm/client.py` are still stubs, so `route()` only covers the retrieval half described in the module's original docstring.

Wired [`api/routes/query.py`](src/memora/api/routes/query.py): `POST /query/` takes `{"query": "..."}`, runs `route()`, returns `{"results": [{"text", "source", "chunk_index", "score"}, ...]}`.

- Pulled the `get_vector_store()` singleton out of `ingest.py` into a new [`api/dependencies.py`](src/memora/api/dependencies.py) so both routes share one `VectorStore` instance instead of each route opening its own `lru_cache`'d connection to the same on-disk path. `ingest.py` now imports it from there — existing tests importing `get_vector_store` from `memora.api.routes.ingest` still work since the name is re-exported by that import.
- Tests: [`tests/test_router.py`](tests/test_router.py) (empty store, relevance ordering, `top_k`/`candidate_k`) and [`tests/test_query_route.py`](tests/test_query_route.py) (empty-store response, ingest-then-query round trip through `TestClient`). Full suite: 35/35 passing.

This completes a working ingest → query round trip end to end: `POST /ingest/` → `POST /query/`.

**Next steps:** implement `memory/context_builder.py` (assemble reranked chunks into an LLM prompt within a token budget) and `llm/client.py` (Ollama wrapper), then extend `router.route()` (or a new `answer()`) to call through to the LLM and return a real answer instead of raw ranked chunks. `observability/tracer.py` should start recording traces around the same time, per the original module docstring.

## 2026-09-13 (cont.) — Context builder, LLM client, tracing, and `answer()` implemented

This completes the query half of the pipeline end to end, matching the original `query.py` stub comment: hybrid retrieval → rerank → context builder → LLM, returning an answer plus sources and a trace id.

- [`memory/context_builder.py`](src/memora/memory/context_builder.py): `build_context(memories, token_budget)` packs `RerankedChunk`s (most relevant first, from `reranker.rerank()`) into `"[source]\ntext"` blocks joined by blank lines, skipping any chunk that would push the running total over budget (rather than stopping at the first oversized one, so a smaller lower-ranked chunk can still fit). No tokenizer dependency yet, so token cost is approximated at ~4 chars/token — same character-based tradeoff `ingestion/chunker.py` already made.
- [`llm/client.py`](src/memora/llm/client.py): `generate(prompt, model_name=None)` wraps `ollama.Client.generate()`, lazily constructed and `lru_cache`'d the same way `ingestion/embedder.py` caches its model. No Ollama server is running in this dev environment, so this is exercised only via a monkeypatched fake client in tests, not a live call — untested against a real model.
- [`observability/store.py`](src/memora/observability/store.py): implemented `ObservabilityStore` (was a stub) — plain SQLite, one `traces` table, JSON-serialized `retrieved`/`reranked` columns. Opens and closes a fresh connection per call rather than holding one open, since the store is a long-lived singleton that FastAPI's threadpool can hit from different threads for sync endpoints — sidesteps `sqlite3`'s not-thread-safe-by-default connections without needing locks.
- [`observability/tracer.py`](src/memora/observability/tracer.py): `record_trace(query, retrieved, reranked, context, response, store)` builds a `Trace` (uuid4 id, UTC timestamp) and saves it in one shot, returning the trace id. Deliberately not the stub's original `start_trace(query) -> str` shape — the whole pipeline runs synchronously in one request, so there's no need for a multi-call incremental builder; one call with all the stage data already in hand is simpler. Conflict detection isn't recorded (needs `knowledge_graph/`, still a stub).
- [`retrieval/router.py`](src/memora/retrieval/router.py): added `answer(query, store, observability_store, candidate_k=20, top_k=5, token_budget=2000) -> AnswerResult` (`answer`, `sources`, `trace_id`) alongside the existing `route()` — `route()` still returns raw ranked chunks (useful on its own, e.g. for a future search-only endpoint), `answer()` builds on it: `retrieve` → `rerank` → `build_context` → a fixed prompt template → `generate` → `record_trace`.
- [`api/dependencies.py`](src/memora/api/dependencies.py): pulled `get_vector_store()` out of `ingest.py` (still re-exported from there so existing test imports keep working) and added `get_observability_store()` alongside it, both `lru_cache` singletons.
- [`api/routes/query.py`](src/memora/api/routes/query.py): `POST /query/` now calls `router.answer()` and returns `{"answer", "sources", "trace_id"}` instead of raw ranked chunks.
- Tests: [`tests/test_context_builder.py`](tests/test_context_builder.py), [`tests/test_llm_client.py`](tests/test_llm_client.py) (fake `ollama.Client` via monkeypatch — no real Ollama server needed), [`tests/test_observability_store.py`](tests/test_observability_store.py), [`tests/test_tracer.py`](tests/test_tracer.py), new `answer()` cases in [`tests/test_router.py`](tests/test_router.py), and an updated [`tests/test_query_route.py`](tests/test_query_route.py) (all monkeypatch `router.generate` to avoid needing a live LLM). Full suite: 50/50 passing.

**Next steps:** get a real Ollama instance running locally (`ollama pull llama3.2:3b`) and manually verify `POST /query/` end to end against it — everything so far has only been tested against a faked LLM client. After that: `knowledge_graph/` (entity/relationship extraction + graph store) to add the third retrieval signal and enable conflict detection in traces, and `observability/metrics.py` + the `GET /observability/*` routes to surface the traces this now records.

## 2026-09-16 — Ollama installed, model pulled, verified live end to end

Ollama wasn't installed on this machine yet (no ROCm support for this laptop's integrated Radeon graphics, so it runs CPU-only via llama.cpp — expected, matches the earlier hardware recommendation). User installed it manually from ollama.com; then pulled the model:

```
ollama pull llama3.2:3b
```

2.0GB download, digest verified. Model weights live in Ollama's own shared store (`%LOCALAPPDATA%`-adjacent `~/.ollama/models` on this machine), **not** in the repo — same reasoning as why `data/db/` is gitignored, just one level further out: Ollama is a system-wide service other projects can reuse, and the repo only needs to reference the model by name (`LLM_MODEL=llama3.2:3b` in `.env`) over Ollama's HTTP API, never the weight files directly.

Ran a manual (non-pytest) round trip against the live server, bypassing the mocked `generate()` the test suite uses: `ingest_source()` on a one-line note about Memora's own RRF-based retrieval, then `retrieval.router.answer()` on a question about it. Got back a correct, grounded answer citing the right source file and chunk index, plus a trace id — confirms `llm/client.py`'s Ollama wrapper, and the full parser → chunker → embedder → vector_store → hybrid retrieval → rerank → context_builder → LLM → tracer chain, all work end to end against real infrastructure, not just mocks. Full pytest suite re-run afterward to confirm no regressions: 50/50 still passing (those remain mocked — the live check was a separate one-off script, not added as a test, since it depends on Ollama actually running).

**Next steps:** `knowledge_graph/` (entity/relationship extraction + graph store) to add the third retrieval signal and enable conflict detection in traces, then `observability/metrics.py` + `GET /observability/*` routes to surface the traces already being recorded.

## 2026-09-16 (cont.) — Knowledge graph, conflict detection, and observability metrics/routes implemented

No new tool/service needed for the graph — same call as the original stack decision: `networkx` (already a pip dependency) persisted to SQLite, same embedded pattern as `observability/store.py`. The optional `graph-db` (Neo4j) extra in `pyproject.toml` stays there unused unless this ever needs real graph-query complexity.

- [`knowledge_graph/extractor.py`](src/memora/knowledge_graph/extractor.py): `extract(text) -> list[Relationship]` (subject/relation/object), LLM-based via `llm.client.generate()`, parsing a `subject | relation | object` per-line format rather than JSON — much more reliably parseable out of a 3B model than asking it for valid JSON. **Tuning note from testing against the real model**: the first prompt draft (multi-line instructions, a parenthetical example, "explicitly stated") made `llama3.2:3b` reply "there is no text provided" even though the text was right there in the prompt — a small model got lost in its own instructions. Cut down to one plain paragraph ("Text: {text}\n\nList the factual relationships... Output only the triples...") and it started extracting cleanly. Verified against the live model, not just the mocked test suite.
- [`knowledge_graph/graph_store.py`](src/memora/knowledge_graph/graph_store.py): `GraphStore(path)` loads a `networkx.MultiDiGraph` from a SQLite `relationships` table at startup (same per-call-connection pattern as `ObservabilityStore`) and mirrors every `add()` to both. Three methods: `add(relationships, source, chunk_index)`; `related_chunks(query, hops=1)` — matches graph nodes appearing as whole words/phrases in the query text (regex word-boundary match, case-insensitive; no NER on queries, just literal containment - covers the common case for a personal KB), BFS-expands `hops` steps, and ranks `(source, chunk_index)` pairs by how many matched edges touch them; `conflicts(subjects=None)` — subject+relation pairs recorded with more than one distinct object, optionally scoped to a subject set.
- **Conflict detection is a documented, deliberately coarse heuristic** (string identity on `(subject, relation)`, not real contradiction detection) - "Memora | uses | LanceDB" + "Memora | uses | Neo4j" would flag exactly like "Alice | lives in | Paris" + "Alice | lives in | Tokyo" would, even though only one of those is an actual contradiction. Telling them apart needs semantic understanding of the relation, out of scope here. **This showed up immediately in live testing**: extracting from one real note produced `Memora | is | Memora` and `Memora | is | a local-first personal knowledge engine` as separate triples (3B-model imprecision - "Memora" got extracted as both subject and object of its own "is" relation), which the heuristic correctly-per-its-rules but incorrectly-per-reality flagged as a conflict. Recorded here as a known limitation rather than tuned away, since fixing it properly means the entailment-based approach this heuristic was explicitly built to avoid for now.
- Wired end to end:
  - [`ingestion/pipeline.py`](src/memora/ingestion/pipeline.py): `ingest_source(source_path, store, graph_store=None)` - when a `GraphStore` is passed, runs `extract()` per chunk and adds any relationships, tagged with that chunk's `(source, chunk_index)`. Optional and defaults to `None` so the two original pipeline tests (and anything else not passing a graph store) don't suddenly need a live LLM.
  - [`retrieval/hybrid.py`](src/memora/retrieval/hybrid.py): `retrieve(..., graph_store=None)` - the old two-list RRF fusion loop generalized to `list[list[...]]` so adding the graph-ranked list (chunks reachable from query-matched entities) was a ~5-line change, not a rewrite. Skipped entirely when `graph_store` is `None` or matches nothing, so existing behavior is unchanged by default.
  - [`retrieval/router.py`](src/memora/retrieval/router.py): `route()`/`answer()` both take `graph_store=None` and thread it into `retrieve()`; `answer()` additionally computes `graph_store.conflicts(set(graph_store.match_entities(query)))` - conflicts scoped to entities the query actually touched, not a full-graph dump on every trace - and passes that into `record_trace`.
  - [`observability/store.py`](src/memora/observability/store.py): `Trace` gained a `conflicts: list[dict]` field and the schema a matching column; added `list_traces()` (ordered newest-first) for `metrics.py` to aggregate over. No existing local `data/db/observability.sqlite3` to migrate - never got created outside test tmp dirs and the earlier live-check script's own tmp dir.
  - [`observability/tracer.py`](src/memora/observability/tracer.py): `record_trace(..., conflicts=None)` - matches the docstring's original "query -> retrieved -> scores -> reranking -> conflicts -> selected -> context -> response" list, which was one stage short of true until now.
  - [`memory/context_builder.py`](src/memora/memory/context_builder.py): `_estimate_tokens` → public `estimate_tokens`, so `metrics.py` could reuse the same ~4-chars/token approximation instead of duplicating it.
  - [`api/dependencies.py`](src/memora/api/dependencies.py) / [`api/routes/ingest.py`](src/memora/api/routes/ingest.py) / [`api/routes/query.py`](src/memora/api/routes/query.py): added `get_graph_store()` singleton, injected into both routes so `POST /ingest/` always extracts relationships and `POST /query/` always gets the graph retrieval signal + query-scoped conflicts.
- [`observability/metrics.py`](src/memora/observability/metrics.py): `compute_metrics(store) -> Metrics` (trace_count, avg_sources_per_trace, avg_context_tokens, contradiction_rate, unique_sources_cited) - **deliberately limited to what a trace actually records**. The README's other targets (retrieval relevance/recall, duplicate memories, memory freshness, answer grounding) need labeled ground truth or an entailment/NLI check that doesn't exist here yet, so rather than fake plausible-looking numbers for those, they're left out and the module docstring says so explicitly.
- [`api/routes/observability.py`](src/memora/api/routes/observability.py): `GET /observability/traces/{trace_id}` (404 if missing) and `GET /observability/metrics`, both reusing the existing `get_observability_store` dependency - no new wiring needed there since it already existed for the query route.

**Verified against the live Ollama model** (not just the mocked test suite, same as the earlier ingest/query milestone): ingested a real note with graph extraction on, confirmed `graph_store.related_chunks("Memora")` found the right chunk, ran a query that got the correct answer ("LanceDB.") with the graph signal contributing to retrieval, and pulled real (if noisy, per the conflict-detection note above) numbers back from `compute_metrics()`.

Tests: [`tests/test_extractor.py`](tests/test_extractor.py), [`tests/test_graph_store.py`](tests/test_graph_store.py), [`tests/test_metrics.py`](tests/test_metrics.py), [`tests/test_observability_route.py`](tests/test_observability_route.py) (new), plus new cases added to `test_pipeline.py`, `test_hybrid.py`, `test_router.py`, `test_tracer.py`, `test_observability_store.py`. `test_ingest_route.py` and `test_query_route.py` fixtures updated to override `get_graph_store` (isolated `tmp_path` graph, same as the vector/observability stores) and, for the ingest route, monkeypatch `pipeline.extract` so route tests don't need a live LLM. Full suite: 77/77 passing.

**Next steps:** the README's remaining "Memory Quality Evaluation" targets that `metrics.py` deliberately didn't implement (relevance/recall against ground truth, duplicate detection, freshness, answer grounding) would be the natural next layer, but need design work first (what's the ground truth, what counts as a duplicate). Shorter-term and lower-risk: a basic UI or CLI to actually use `POST /ingest`, `POST /query`, and the new `GET /observability/*` routes interactively, since everything so far has been exercised through pytest and one-off scripts.
