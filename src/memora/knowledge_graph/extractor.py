"""Pulls concepts/entities/relationships out of a chunk (LLM-based) for the
conceptual-memory graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from memora.llm.client import generate

_EXTRACTION_PROMPT = (
    "Text: {text}\n\n"
    "List the factual relationships stated in the text above as subject | relation | object, one per line. "
    "Reuse the same short name for the same entity throughout. "
    "Output only the triples, nothing else. If there are no clear relationships, output nothing."
)


@dataclass(frozen=True)
class Relationship:
    subject: str
    relation: str
    object: str


def extract(text: str) -> list[Relationship]:
    response = generate(_EXTRACTION_PROMPT.format(text=text))
    return _parse(response)


def _parse(response: str) -> list[Relationship]:
    relationships = []
    for line in response.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        subject, relation, obj = parts
        relationships.append(Relationship(subject=subject, relation=relation, object=obj))
    return relationships
