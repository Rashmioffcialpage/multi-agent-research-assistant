from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    max_iterations: int = Field(default=2, ge=1, le=5)


class SourceOut(BaseModel):
    doc_id: str
    heading: str


class ResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    answer: str
    sources: list[SourceOut]
    iterations: int
