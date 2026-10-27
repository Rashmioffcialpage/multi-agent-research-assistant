import pytest
from fastapi.testclient import TestClient

from research_assistant.api.app import app, get_llm, get_retriever
from research_assistant.llm.fake_provider import FakeLLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def client(retriever: BM25Retriever):
    llm = FakeLLM(
        responses={
            "ROLE: planner": '["what is model quantization"]',
            "ROLE: critic": "SUFFICIENT",
            "ROLE: writer": "Quantization reduces numeric precision to shrink models [1].",
        }
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_retriever] = lambda: retriever
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_research_returns_answer_and_sources(client: TestClient):
    r = client.post("/research", json={"question": "What is model quantization?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Quantization reduces numeric precision to shrink models [1]."
    assert body["sub_questions"] == ["what is model quantization"]
    assert len(body["sources"]) > 0
    assert body["iterations"] == 1


def test_research_rejects_empty_question(client: TestClient):
    r = client.post("/research", json={"question": ""})
    assert r.status_code == 422


def test_research_rejects_out_of_range_max_iterations(client: TestClient):
    r = client.post("/research", json={"question": "q", "max_iterations": 99})
    assert r.status_code == 422
