from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class EvidenceItem(TypedDict):
    sub_question: str
    passages: list[dict]  # serialized retrieval.bm25_retriever.Passage


class Source(TypedDict):
    doc_id: str
    heading: str


class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    # Annotated with operator.add: each node returns only the *new* evidence
    # it gathered, and LangGraph appends it to what's already in state,
    # rather than the node having to remember and re-return prior evidence.
    evidence: Annotated[list[EvidenceItem], operator.add]
    critic_verdict: str  # "" | "SUFFICIENT" | "INSUFFICIENT"
    critic_feedback: str
    iterations: int
    max_iterations: int
    final_answer: str
    sources: list[Source]
