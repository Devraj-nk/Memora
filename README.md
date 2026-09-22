# Memora

Personal Knowledge Engine with an Observability Platform for the Memory Layer

A local-first personal knowledge system that continuously converts a user's documents, code, notes, and web content into an evidence-grounded knowledge base, while providing observability into how its memory is retrieved, ranked, updated, contradicted, and used by the LLM.

Personal Knowledge Engine — builds and retrieves the user's knowledge.
Memory Observability Layer — explains and evaluates what the system remembered and why it used that memory.

## Objectives

1) **Structured Knowledge Representation**

Maintain both:
- Semantic memory — Documents → Chunks → Embeddings
- Conceptual memory — Concepts → Entities → Relationships → Sources

This gives us both vector retrieval and graph-based reasoning.

2) **Evidence-Grounded Retrieval**

Given a query, retrieve information using multiple signals — keyword + semantic similarity + knowledge graph + metadata — and rerank the resulting candidates before sending them to the LLM.

3) **Memory Observability**

Record what happened during every retrieval/response cycle: query → retrieved memories → scores → reranking → conflicts → selected memories → LLM context → response. This lets the user/developer inspect why the system produced an answer.

4) **Memory Quality Evaluation**

Measure retrieval relevance, retrieval recall, memory freshness, contradiction rate, duplicate memories, context/token usage, and answer grounding. (Source quality and true retrieval relevance/recall against a labeled query set are left as future work — see [Implementation notes](#implementation-notes--known-limitations).)

## Tech stack

Python 3.11+, `src`-layout package `memora`. Everything runs locally — no cloud services, no server processes to manage beyond Ollama.

| Concern | Choice | Why |
|---|---|---|
| API | FastAPI + uvicorn | async-friendly, typed request/response models |
| CLI | `argparse` + `httpx` | talks HTTP to the running API, same as any other client |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` | small/fast local default |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder relevance scoring over retrieval candidates |
| Grounding check | `cross-encoder/nli-deberta-v3-xsmall` | NLI entailment model, separate from the answering LLM |
| Vector store | LanceDB | embedded, no server process |
| Keyword retrieval | `rank-bm25` | classic sparse signal, fused with vector search via RRF |
| Knowledge graph | `networkx` (in-process), persisted to SQLite | entities/relationships extracted from ingested text |
| LLM / SLM | Ollama, default model `llama3.2:3b` | local-first; swap via `.env` (`LLM_MODEL`) for quality/speed tradeoffs |
| Observability store | SQLite | one row per retrieval/response trace |
| Config | `pydantic-settings` (`.env` file) | see [`.env.example`](.env.example) / [`config.py`](src/memora/config.py) |

Optional extras (`pyproject.toml`):
- `parsing` — `unstructured[all-docs]` + `pytesseract` for PDFs/OCR (needs Tesseract + Poppler installed separately). Not required for plain text/markdown/code.
- `graph-db` — `neo4j`, only if the knowledge graph outgrows in-process `networkx` + SQLite.
- `dev` — `pytest`.

LanceDB and the observability/graph SQLite files are all embedded — nothing to run except the app itself and Ollama.

## High-level design

```
                         ┌─────────────────────┐
                         │     User / UI       │
                         └──────────┬──────────┘
                                    │
                              Query / Upload
                                    │
                 ┌──────────────────┴─────────────────┐
                 │                                    │
                 ▼                                    ▼
        ┌─────────────────┐                  ┌─────────────────┐
        │ Ingestion       │                  │ Query / Memory  │
        │ Pipeline        │                  │ Router          │
        └────────┬────────┘                  └────────┬────────┘
                 │                                    │
                 ▼                                    ▼
        ┌─────────────────┐                  ┌─────────────────┐
        │ Parser          │                  │ Hybrid Retrieval│
        └────────┬────────┘                  └────────┬────────┘
                 │                                    │
                 ▼                                    │
        ┌─────────────────┐                           │
        │ Chunker         │                           │
        └────────┬────────┘                           │
                 │                                    │
                 ▼                                    ▼
        ┌─────────────────┐                  ┌─────────────────┐
        │ Embedding       │                  │ Candidate       │
        │ Generator       │                  │ Memories        │
        └────────┬────────┘                  └────────┬────────┘
                 │                                    │
                 ▼                                    ▼
        ┌─────────────────┐                  ┌─────────────────┐
        │ Vector Store    │◄─────────────────│ Reranker        │
        └─────────────────┘                  └────────┬────────┘
                 │                                    │
                 │                                    ▼
                 │                            ┌─────────────────┐
                 └───────────────────────────►│ Knowledge Graph │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Context Builder │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Local SLM / LLM │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Answer + Sources│
                                              └────────┬────────┘
                                                       │
                                                       ▼
                              ┌────────────────────────────────────┐
                              │     Memory Observability Layer     │
                              │                                    │
                              │ Retrieval traces                   │
                              │ relevance scores                   │
                              │ memory age                         │
                              │ conflicts                          │
                              │ confidence                         │
                              │ token/context cost                 │
                              │ response attribution               │
                              └────────────────────────────────────┘
```

### Data flow

**Ingestion** (`POST /ingest/`, [`ingestion/pipeline.py`](src/memora/ingestion/pipeline.py)):

```
file path → parser.parse()        → ParsedDocument (text, source, mime_type, modified_at)
          → chunker.chunk()       → list[Chunk] (sliding window, whitespace-aware, overlap)
          → embedder.embed()      → one L2-normalized vector per chunk
          → vector_store.add()    → persisted to LanceDB
          → extractor.extract()   → per chunk, LLM-derived (subject, relation, object) triples
          → graph_store.add()     → persisted to a networkx graph backed by SQLite
```

**Query** (`POST /query/`, [`retrieval/router.py`](src/memora/retrieval/router.py) `answer()`):

```
query → hybrid.retrieve()      → vector search + BM25 + graph-reachable chunks,
                                  fused by reciprocal rank fusion (RRF)
      → reranker.rerank()      → cross-encoder relevance scoring, most-relevant first
      → context_builder.build_context() → pack top chunks into a token-budgeted prompt
      → llm.client.generate()  → Ollama call, produces the answer
      → tracer.record_trace()  → persists query/retrieved/reranked/conflicts/context/
                                  response as one row in the observability store
→ {answer, sources, trace_id}
```

**Observability** (`GET /observability/*`, [`observability/metrics.py`](src/memora/observability/metrics.py)):

```
stored traces → compute_metrics() → trace_count, avg_sources_per_trace, avg_context_tokens,
                                     contradiction_rate, unique_sources_cited,
                                     answer_grounding_rate (via a separate NLI model),
                                     avg_memory_age_days, duplicate_chunk_pairs
                                     (cosine similarity over stored embeddings)

sampled chunks → evaluate_retrieval() → LLM generates a synthetic question per chunk,
                                         checks whether retrieval finds that chunk again
                                         → recall_at_k (a regression signal, not an
                                           absolute score — see limitations below)
```

## Project layout

```
src/memora/
  api/
    main.py            FastAPI app, mounts the three routers below
    dependencies.py     lru_cache singletons: VectorStore, GraphStore, ObservabilityStore
    routes/
      ingest.py          POST /ingest/
      query.py           POST /query/
      observability.py   GET /observability/traces/{id}, GET /observability/metrics,
                          POST /observability/evaluate
  ingestion/
    parser.py            file → ParsedDocument (text/markdown/code by extension allowlist)
    chunker.py            ParsedDocument → list[Chunk] (sliding window, overlap)
    embedder.py           text → embedding vectors (sentence-transformers)
    pipeline.py            parser → chunker → embedder → vector_store (+ optional graph extraction)
  retrieval/
    hybrid.py              vector search + BM25 + knowledge-graph signal, fused via RRF
    reranker.py            cross-encoder reranking of candidates
    router.py               route() for raw ranked chunks; answer() for the full retrieve→LLM path
  knowledge_graph/
    extractor.py            LLM-based (subject, relation, object) triple extraction
    graph_store.py           networkx graph persisted to SQLite; related_chunks(), conflicts()
  memory/
    vector_store.py          LanceDB wrapper: add(), search(), all_chunks(), all_chunks_with_vectors()
    context_builder.py       pack reranked chunks into a token-budgeted LLM prompt
  llm/
    client.py                Ollama client wrapper (generate())
  observability/
    store.py                 SQLite-backed trace storage
    tracer.py                builds and records a Trace for one query/response cycle
    metrics.py                compute_metrics() over stored traces + the vector store
    evaluation.py             synthetic retrieval-recall self-check
    grounding.py               NLI-based answer-grounding check (is_grounded())
  cli.py                     `memora` console command — HTTP client for the API above
  config.py                  pydantic-settings, reads .env
```

## Running it

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com), installed and running as a local service
- (Optional) Tesseract + Poppler if you install the `parsing` extra for PDF/OCR ingestion

### Setup

```bash
# from the repo root
pip install -e ".[dev]"

# pull the default local model
ollama pull llama3.2:3b

cp .env.example .env   # adjust model names / paths if needed

demo_files is a local directory for your own notes, code, and documents to ingest. Add a file there to test/work
```

Config is read from `.env` via [`config.py`](src/memora/config.py) — see [`.env.example`](.env.example) for every setting (LLM model/host, embedding/reranker/grounding model names, and on-disk paths for the vector store, graph, and observability DB, all under `data/db/` by default).

### Start the server

```bash
uvicorn memora.api.main:app --reload
```

Open `http://localhost:8000/` for the Memora console. The UI provides three views:

- **Ask memory** — query the full retrieval and answer pipeline, inspect cited sources, and follow the generated trace.
- **Ingest source** — add a local text, Markdown, or code file to the parsing, chunking, embedding, and graph pipeline.
- **Observability** — view memory-quality metrics, inspect the latest retrieval anatomy, and manually run the synthetic recall evaluation.

This mounts:
- `POST /ingest/` — `{"path": "<local file path>"}` → `{"source", "chunk_count"}`
- `POST /query/` — `{"query": "<question>"}` → `{"answer", "sources", "trace_id"}`
- `GET /observability/traces/{trace_id}` — full retrieval/response trace
- `GET /observability/metrics` — aggregate memory-quality metrics
- `POST /observability/evaluate` — `{"top_k", "sample_size"}` → synthetic retrieval-recall check
- `GET /health`

### Use the CLI

The `memora` console command (installed via the `pip install -e .` above) is an HTTP client for the server — start `uvicorn` first:

```bash
memora ingest path/to/notes.md
memora query "what vector store does Memora use?"
memora trace <trace_id>
memora metrics
memora evaluate --top-k 5 --sample-size 10
```

### Tests

```bash
pytest tests/
```

Tests that hit a real downloaded model (embedder/reranker/grounding) are marked `slow` and mocked by default elsewhere in the suite:

```bash
pytest tests/ -m "not slow"   # skip the real-model tests
```

## Implementation notes / known limitations

- **Conflict detection** is a deliberately coarse heuristic: any `(subject, relation)` pair recorded with more than one distinct object is flagged, with no semantic understanding of whether the objects actually contradict each other (e.g. "Memora uses LanceDB" + "Memora uses Neo4j" is flagged the same way "Alice lives in Paris" + "Alice lives in Tokyo" would be).
- **Retrieval recall** (`POST /observability/evaluate`) is a synthetic, self-generated check — there's no labeled query set for a personal knowledge base, so questions are generated by the LLM directly from sampled chunks. It's optimistic by construction and intended as a regression signal, not an absolute score.
- **Answer grounding** uses a dedicated NLI entailment model (not the same LLM that generated the answer), scored against the context minus its `[source]` header line (stripped — file-path-like headers were found to dilute the NLI premise).
- **Source quality** is not implemented — the README's original objective gives no concrete rubric to build one against.
- Full development history, including issues found while testing against real (non-mocked) models, is in [`work.md`](work.md).
