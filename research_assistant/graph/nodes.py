"""The four agent nodes: planner, researcher, critic, writer.

Every node function has the signature (state, **deps) -> dict — LangGraph
merges the returned dict into the running ResearchState (see
research_assistant.graph.state). Dependencies (the LLM, the retriever) are
bound via functools.partial when the graph is built (see build.py), not
imported globally, so the exact same node functions run against a real
OllamaLLM or a scripted FakeLLM without any code change.
"""

from __future__ import annotations

import json
import re

from research_assistant.graph.state import ResearchState
from research_assistant.llm.base import LLM
from research_assistant.retrieval.bm25_retriever import BM25Retriever
from research_assistant.tools.corpus_search import search_corpus

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list[str]:
    """Best-effort JSON array extraction from an LLM response.

    Real LLMs sometimes wrap JSON in prose or a markdown code fence even
    when asked for "only JSON" — this looks for the first [...] span
    rather than requiring the whole response to be valid JSON on its own.
    """
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed if str(item).strip()]


def planner_node(state: ResearchState, *, llm: LLM) -> dict:
    prompt = (
        "ROLE: planner\n\n"
        "Break the following research question into 2-4 focused, independently "
        "searchable sub-questions. Respond with ONLY a JSON array of strings, "
        "nothing else.\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)
    sub_questions = _extract_json_array(response)
    if not sub_questions:
        # Real LLM didn't follow the format, or FakeLLM has no marker for
        # this prompt — fall back to researching the question as-is rather
        # than failing the whole run.
        sub_questions = [state["question"]]
    return {"sub_questions": sub_questions}


def researcher_node(state: ResearchState, *, retriever: BM25Retriever, k: int = 4) -> dict:
    already_researched = {e["sub_question"] for e in state["evidence"]}
    todo = [sq for sq in state["sub_questions"] if sq not in already_researched]

    new_evidence = []
    for sub_question in todo:
        passages = search_corpus(retriever, sub_question, k=k)
        new_evidence.append(
            {
                "sub_question": sub_question,
                "passages": [
                    {"doc_id": p.doc_id, "heading": p.heading, "text": p.text, "score": p.score}
                    for p in passages
                ],
            }
        )
    return {"evidence": new_evidence}


def _render_evidence(evidence: list[dict]) -> str:
    blocks = []
    n = 0
    for item in evidence:
        passage_lines = []
        for p in item["passages"]:
            n += 1
            passage_lines.append(f"  [{n}] ({p['doc_id']} :: {p['heading']})\n  {p['text']}")
        passages_text = (
            "\n\n".join(passage_lines) if passage_lines else "  (no relevant passages found)"
        )
        blocks.append(f"Sub-question: {item['sub_question']}\n{passages_text}")
    return "\n\n".join(blocks) if blocks else "(no evidence gathered yet)"


def critic_node(state: ResearchState, *, llm: LLM) -> dict:
    prompt = (
        "ROLE: critic\n\n"
        "Given the research question and the evidence gathered so far, decide "
        "whether there is enough evidence to write a complete, well-cited "
        "answer. Respond with exactly one line:\n"
        '  "SUFFICIENT" if the evidence is enough, or\n'
        '  "INSUFFICIENT: <one specific follow-up sub-question to research next>" '
        "if it isn't.\n\n"
        f"Question: {state['question']}\n\n"
        f"Evidence so far:\n{_render_evidence(state['evidence'])}"
    )
    response = llm.invoke(prompt).strip()
    iterations = state["iterations"] + 1

    if response.upper().startswith("SUFFICIENT"):
        return {"critic_verdict": "SUFFICIENT", "critic_feedback": "", "iterations": iterations}

    feedback = response.split(":", 1)[1].strip() if ":" in response else response
    new_sub_questions = state["sub_questions"]
    if feedback and feedback not in new_sub_questions:
        new_sub_questions = [*new_sub_questions, feedback]

    return {
        "critic_verdict": "INSUFFICIENT",
        "critic_feedback": feedback,
        "iterations": iterations,
        "sub_questions": new_sub_questions,
    }


def writer_node(state: ResearchState, *, llm: LLM) -> dict:
    prompt = (
        "ROLE: writer\n\n"
        "Write a clear, well-organized answer to the research question using "
        "only the evidence below. Cite passages inline by their [n] number. "
        "If the evidence doesn't fully answer the question, say so explicitly "
        "rather than filling gaps with unstated assumptions.\n\n"
        f"Question: {state['question']}\n\n"
        f"Evidence:\n{_render_evidence(state['evidence'])}"
    )
    answer = llm.invoke(prompt)

    sources = []
    seen = set()
    for item in state["evidence"]:
        for p in item["passages"]:
            key = (p["doc_id"], p["heading"])
            if key not in seen:
                seen.add(key)
                sources.append({"doc_id": p["doc_id"], "heading": p["heading"]})

    return {"final_answer": answer, "sources": sources}
