from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from research_assistant.graph.nodes import critic_node, planner_node, researcher_node, writer_node
from research_assistant.graph.state import ResearchState
from research_assistant.llm.base import LLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever


def route_after_critic(state: ResearchState) -> str:
    if state["critic_verdict"] == "SUFFICIENT" or state["iterations"] >= state["max_iterations"]:
        return "writer"
    return "researcher"


def build_graph(llm: LLM, retriever: BM25Retriever):
    """Planner -> Researcher -> Critic -> (loop back to Researcher, or Writer).

    The critic->researcher edge is what makes this a graph rather than a
    pipeline: an insufficient-evidence verdict adds a follow-up
    sub-question and routes back for another retrieval pass, bounded by
    ResearchState.max_iterations so it can't loop forever.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", partial(planner_node, llm=llm))
    graph.add_node("researcher", partial(researcher_node, retriever=retriever))
    graph.add_node("critic", partial(critic_node, llm=llm))
    graph.add_node("writer", partial(writer_node, llm=llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"researcher": "researcher", "writer": "writer"}
    )
    graph.add_edge("writer", END)

    return graph.compile()


def initial_state(question: str, max_iterations: int = 2) -> ResearchState:
    return ResearchState(
        question=question,
        sub_questions=[],
        evidence=[],
        critic_verdict="",
        critic_feedback="",
        iterations=0,
        max_iterations=max_iterations,
        final_answer="",
        sources=[],
    )
