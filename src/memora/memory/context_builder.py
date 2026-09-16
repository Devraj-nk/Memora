"""Assembles the final LLM context from reranked memories, tracking
token/context cost for the observability layer.
"""

from __future__ import annotations

from memora.retrieval.reranker import RerankedChunk

# No tokenizer dependency yet, so approximate at ~4 chars/token (a common
# rule of thumb for English text) - chunker.py made the same character-based
# tradeoff for the same reason, see its module notes in work.md.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def build_context(memories: list[RerankedChunk], token_budget: int) -> str:
    sections: list[str] = []
    used_tokens = 0

    for memory in memories:
        block = f"[{memory.source}]\n{memory.text}"
        cost = estimate_tokens(block)
        if used_tokens + cost > token_budget:
            continue
        sections.append(block)
        used_tokens += cost

    return "\n\n".join(sections)
