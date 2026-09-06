# Splits parsed text into retrieval-sized chunks, preserving enough
# metadata (source, offset) to trace a chunk back to its origin.


def chunk(text: str) -> list[str]:
    raise NotImplementedError
