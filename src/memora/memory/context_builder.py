# Assembles the final LLM context from reranked memories, tracking
# token/context cost for the observability layer.


def build_context(memories: list[dict], token_budget: int) -> str:
    raise NotImplementedError
