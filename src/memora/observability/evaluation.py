"""Retrieval relevance/recall, measured against a synthetic ground truth:
for each sampled chunk, ask the LLM for a question that chunk answers, then
check whether retrieval finds that same chunk again for its own question.

There's no external labeled query set for a personal, local knowledge base
to evaluate against, so this manufactures one instead. It's optimistic by
construction - the question is generated directly from the chunk, so it's
an easier match than a real user's query would be - so treat recall_at_k as
a regression signal over time (did this ingestion/retrieval change make
things worse?), not an absolute quality score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from memora.llm.client import generate
from memora.memory.vector_store import VectorStore
from memora.retrieval.hybrid import retrieve

DEFAULT_SAMPLE_SIZE = 10
DEFAULT_TOP_K = 5

_QUESTION_PROMPT = "Text: {text}\n\nWrite one short question that this text answers. Output only the question."


@dataclass(frozen=True)
class RetrievalEvalResult:
    recall_at_k: float
    sample_size: int
    top_k: int


def generate_synthetic_question(text: str) -> str:
    return generate(_QUESTION_PROMPT.format(text=text)).strip()


def evaluate_retrieval(
    store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> RetrievalEvalResult:
    chunks = store.all_chunks()
    if not chunks:
        return RetrievalEvalResult(recall_at_k=0.0, sample_size=0, top_k=top_k)

    rng = random.Random(seed)
    sample = rng.sample(chunks, min(sample_size, len(chunks)))

    hits = 0
    for c in sample:
        question = generate_synthetic_question(c.text)
        results = retrieve(question, store, top_k=top_k)
        if any(r.source == c.source and r.chunk_index == c.chunk_index for r in results):
            hits += 1

    return RetrievalEvalResult(recall_at_k=hits / len(sample), sample_size=len(sample), top_k=top_k)
