from research_assistant.graph.build import initial_state
from research_assistant.graph.nodes import critic_node, planner_node, researcher_node, writer_node
from research_assistant.llm.fake_provider import FakeLLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever


def test_planner_parses_json_array_response():
    llm = FakeLLM(responses={"ROLE: planner": '["sub q one", "sub q two"]'})
    state = initial_state("original question")
    update = planner_node(state, llm=llm)
    assert update["sub_questions"] == ["sub q one", "sub q two"]


def test_planner_falls_back_to_original_question_on_unparseable_response():
    llm = FakeLLM(responses={"ROLE: planner": "I cannot help with that."})
    state = initial_state("original question")
    update = planner_node(state, llm=llm)
    assert update["sub_questions"] == ["original question"]


def test_planner_handles_json_wrapped_in_prose():
    llm = FakeLLM(responses={"ROLE: planner": 'Sure, here you go:\n["a", "b"]\nHope that helps!'})
    state = initial_state("q")
    update = planner_node(state, llm=llm)
    assert update["sub_questions"] == ["a", "b"]


def test_researcher_only_researches_unresearched_sub_questions(retriever: BM25Retriever):
    state = initial_state("q")
    state["sub_questions"] = ["what is model quantization"]
    state["evidence"] = [{"sub_question": "what is model quantization", "passages": []}]

    update = researcher_node(state, retriever=retriever)
    assert update["evidence"] == []  # already researched, nothing new to add


def test_researcher_retrieves_real_passages_for_new_sub_questions(retriever: BM25Retriever):
    state = initial_state("q")
    state["sub_questions"] = ["what is model quantization"]
    state["evidence"] = []

    update = researcher_node(state, retriever=retriever)
    assert len(update["evidence"]) == 1
    assert update["evidence"][0]["sub_question"] == "what is model quantization"
    assert len(update["evidence"][0]["passages"]) > 0
    assert update["evidence"][0]["passages"][0]["doc_id"] == "model-quantization"


def test_critic_sufficient_verdict_stops_the_loop():
    llm = FakeLLM(responses={"ROLE: critic": "SUFFICIENT"})
    state = initial_state("q")
    state["iterations"] = 0
    update = critic_node(state, llm=llm)
    assert update["critic_verdict"] == "SUFFICIENT"
    assert update["iterations"] == 1


def test_critic_insufficient_verdict_appends_a_new_sub_question():
    llm = FakeLLM(responses={"ROLE: critic": "INSUFFICIENT: a follow-up question"})
    state = initial_state("q")
    state["sub_questions"] = ["original sub-question"]
    update = critic_node(state, llm=llm)
    assert update["critic_verdict"] == "INSUFFICIENT"
    assert update["critic_feedback"] == "a follow-up question"
    assert update["sub_questions"] == ["original sub-question", "a follow-up question"]


def test_critic_does_not_duplicate_an_already_present_sub_question():
    llm = FakeLLM(responses={"ROLE: critic": "INSUFFICIENT: dup question"})
    state = initial_state("q")
    state["sub_questions"] = ["dup question"]
    update = critic_node(state, llm=llm)
    assert update["sub_questions"] == ["dup question"]


def test_writer_produces_answer_and_deduplicated_sources():
    llm = FakeLLM(responses={"ROLE: writer": "the synthesized answer"})
    state = initial_state("q")
    state["evidence"] = [
        {
            "sub_question": "sq1",
            "passages": [
                {"doc_id": "doc-a", "heading": "H1", "text": "t1", "score": 1.0},
                {"doc_id": "doc-a", "heading": "H1", "text": "t1", "score": 1.0},  # duplicate
            ],
        },
        {
            "sub_question": "sq2",
            "passages": [{"doc_id": "doc-b", "heading": "H2", "text": "t2", "score": 0.5}],
        },
    ]
    update = writer_node(state, llm=llm)
    assert update["final_answer"] == "the synthesized answer"
    assert update["sources"] == [
        {"doc_id": "doc-a", "heading": "H1"},
        {"doc_id": "doc-b", "heading": "H2"},
    ]
