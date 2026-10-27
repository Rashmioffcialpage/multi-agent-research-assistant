from research_assistant.graph.build import build_graph, initial_state
from research_assistant.llm.fake_provider import FakeLLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever


def test_single_pass_when_critic_immediately_approves(retriever: BM25Retriever):
    llm = FakeLLM(
        responses={
            "ROLE: planner": '["what is model quantization"]',
            "ROLE: critic": "SUFFICIENT",
            "ROLE: writer": "Quantization reduces numeric precision [1].",
        }
    )
    graph = build_graph(llm, retriever)
    result = graph.invoke(initial_state("What is model quantization?"))

    assert result["iterations"] == 1
    assert result["critic_verdict"] == "SUFFICIENT"
    assert result["final_answer"] == "Quantization reduces numeric precision [1]."
    assert len(result["sources"]) > 0


def test_critic_loop_back_gathers_more_evidence_before_writing(retriever: BM25Retriever):
    """The real test of the graph's cyclic edge, not just the linear path:
    critic rejects once, a new sub-question gets researched, then critic
    approves and the writer runs — evidence from BOTH passes must be present.
    """
    llm = FakeLLM(
        responses={
            "ROLE: planner": '["how does BM25 lexical retrieval work"]',
            "ROLE: critic": [
                "INSUFFICIENT: how does hybrid retrieval combine BM25 and dense retrieval",
                "SUFFICIENT",
            ],
            "ROLE: writer": "Answer combining both retrieval passes.",
        }
    )
    graph = build_graph(llm, retriever)
    result = graph.invoke(initial_state("How does retrieval work in RAG?", max_iterations=3))

    assert result["iterations"] == 2
    assert len(result["sub_questions"]) == 2
    # one evidence group per sub-question, from two separate research passes
    assert len(result["evidence"]) == 2
    assert result["critic_verdict"] == "SUFFICIENT"
    assert result["final_answer"] == "Answer combining both retrieval passes."


def test_bounded_by_max_iterations_even_if_critic_never_approves(retriever: BM25Retriever):
    """Guards the failure mode called out in data/corpus/multi-agent-systems.md:
    a critic that never approves must not loop forever.
    """
    llm = FakeLLM(
        responses={
            "ROLE: planner": '["question one"]',
            "ROLE: critic": "INSUFFICIENT: always ask for more",
            "ROLE: writer": "Best-effort answer with limited evidence.",
        }
    )
    graph = build_graph(llm, retriever)
    result = graph.invoke(initial_state("Some question", max_iterations=2))

    assert result["iterations"] == 2  # stopped at max_iterations, not looping forever
    assert result["final_answer"] == "Best-effort answer with limited evidence."


def test_full_llm_call_sequence_is_planner_then_researcher_loop_then_writer(
    retriever: BM25Retriever,
):
    llm = FakeLLM(
        responses={
            "ROLE: planner": '["q1"]',
            "ROLE: critic": ["INSUFFICIENT: q2", "SUFFICIENT"],
            "ROLE: writer": "final",
        }
    )
    graph = build_graph(llm, retriever)
    graph.invoke(initial_state("q", max_iterations=3))

    roles_called = [
        "planner" if "ROLE: planner" in c else "critic" if "ROLE: critic" in c else "writer"
        for c in llm.calls
    ]
    assert roles_called == ["planner", "critic", "critic", "writer"]
