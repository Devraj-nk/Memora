# Orchestrates a query end to end: hybrid retrieval -> rerank -> context
# builder -> LLM, emitting an observability trace at each stage.


def answer(query: str) -> dict:
    raise NotImplementedError
