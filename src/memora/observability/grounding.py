"""Checks whether a generated answer is actually supported by its context,
using a small NLI (entailment) model rather than asking the same generative
LLM to judge its own answer.

Deliberately a separate, dedicated model - same "small specialized model"
pattern already used for retrieval.reranker's cross-encoder - rather than
another round trip through llm.client, which would be both slower and a
noisier signal (a 3B model judging its own output).
"""

from __future__ import annotations

import re
from functools import lru_cache

from sentence_transformers import CrossEncoder

from memora.config import settings

# memory.context_builder.build_context() prefixes each block with
# "[source]\n" for the LLM prompt's benefit. Verified against the real NLI
# model that leaving it in throws off the premise: the same context+answer
# pair that correctly scored as "entailment" dropped to "neutral" once a
# source path like "[C:\...\note.md]\n" was prepended - the model has never
# seen anything like a file path in training and it appears to dilute the
# premise's signal. Stripped here since it's presentational, not factual
# content.
_SOURCE_HEADER = re.compile(r"^\[.+\]\n", flags=re.MULTILINE)


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def is_grounded(context: str, query: str, answer: str, model_name: str | None = None) -> bool:
    """True if `context` entails `query + answer` per the NLI model's top label.

    An empty context can't ground anything, so that's treated as ungrounded
    without spending a model call on it. The query is prepended to the
    answer to form the NLI hypothesis - not just the answer alone - because
    this NLI model is trained on full-sentence hypotheses, and local LLMs
    routinely answer in a bare fragment ("LanceDB." rather than "Memora
    uses LanceDB."). Verified against the real model: a fragment answer on
    its own got misclassified as "contradiction" regardless of whether it
    was actually correct; prefixing the question fixed it.
    """
    if not context.strip():
        return False

    model = _get_model(model_name or settings.grounding_model)
    premise = _SOURCE_HEADER.sub("", context)
    hypothesis = f"{query} {answer}"
    scores = model.predict([(premise, hypothesis)])[0]
    predicted_label = model.config.id2label[int(scores.argmax())]
    return predicted_label == "entailment"
