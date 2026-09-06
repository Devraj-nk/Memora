# Records one retrieval/response cycle (query -> retrieved -> scores ->
# reranking -> conflicts -> selected -> context -> response) as a trace.


def start_trace(query: str) -> str:
    raise NotImplementedError
