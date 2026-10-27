from pathlib import Path

import pytest

from research_assistant.retrieval.bm25_retriever import BM25Retriever

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture(scope="session")
def retriever() -> BM25Retriever:
    return BM25Retriever(CORPUS_DIR)
