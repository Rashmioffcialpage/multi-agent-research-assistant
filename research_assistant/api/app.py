"""FastAPI serving layer.

Run with: uvicorn research_assistant.api.app:app --reload

get_llm/get_retriever are FastAPI dependencies specifically so tests
(tests/test_api.py) can override them with a FakeLLM/small test corpus via
app.dependency_overrides, without touching this module or needing a real
Ollama server running.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI

from research_assistant.api.schemas import ResearchRequest, ResearchResponse, SourceOut
from research_assistant.graph.build import build_graph, initial_state
from research_assistant.llm.base import LLM
from research_assistant.llm.ollama_provider import OllamaLLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Planner/Researcher/Critic/Writer agents over a local BM25-retrieved corpus.",
)

_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@lru_cache
def get_retriever() -> BM25Retriever:
    corpus_dir = Path(os.environ.get("CORPUS_DIR", _DEFAULT_CORPUS_DIR))
    return BM25Retriever(corpus_dir)


@lru_cache
def get_llm() -> LLM:
    return OllamaLLM(
        model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(
    req: ResearchRequest,
    llm: LLM = Depends(get_llm),
    retriever: BM25Retriever = Depends(get_retriever),
) -> ResearchResponse:
    graph = build_graph(llm, retriever)
    state = initial_state(req.question, max_iterations=req.max_iterations)
    result = graph.invoke(state)

    return ResearchResponse(
        question=result["question"],
        sub_questions=result["sub_questions"],
        answer=result["final_answer"],
        sources=[SourceOut(**s) for s in result["sources"]],
        iterations=result["iterations"],
    )
