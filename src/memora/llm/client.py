"""Thin wrapper over the `ollama` client, pointed at
memora.config.settings.ollama_host / llm_model. Swapping providers means
swapping this module, not the callers.
"""

from __future__ import annotations

from functools import lru_cache

import ollama

from memora.config import settings


@lru_cache(maxsize=1)
def _get_client(host: str) -> ollama.Client:
    return ollama.Client(host=host)


def generate(prompt: str, model_name: str | None = None) -> str:
    client = _get_client(settings.ollama_host)
    response = client.generate(model=model_name or settings.llm_model, prompt=prompt)
    return response.response
