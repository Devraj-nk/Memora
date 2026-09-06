# Combines keyword (rank-bm25), semantic (vector store), and knowledge-graph
# signals into a single candidate set for the reranker.


def retrieve(query: str) -> list[dict]:
    raise NotImplementedError
