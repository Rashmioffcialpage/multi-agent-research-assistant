"""Thin tool wrapper around BM25Retriever for use inside agent nodes."""

from __future__ import annotations

from research_assistant.retrieval.bm25_retriever import BM25Retriever, Passage


def search_corpus(retriever: BM25Retriever, query: str, k: int = 4) -> list[Passage]:
    return retriever.search(query, k=k)


def format_passages(passages: list[Passage]) -> str:
    """Renders passages as numbered, citable blocks for an LLM prompt."""
    if not passages:
        return "(no relevant passages found)"
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"[{i}] ({p.doc_id} :: {p.heading})\n{p.text}")
    return "\n\n".join(blocks)
