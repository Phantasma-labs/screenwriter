# src/graph/nodes/review_shared.py
from __future__ import annotations

from pydantic import BaseModel, Field

REVIEW_RESPONSE_INSTRUCTION = (
    'Respond with JSON only: {"score": <0-10 float>, "passed": <bool>, '
    '"critique": <string>, "actionable_revisions": [<string>, ...]}'
)


class ReviewFeedback(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    passed: bool
    critique: str
    actionable_revisions: list[str] = Field(default_factory=list)


FALLBACK_REVIEW_FEEDBACK = ReviewFeedback(
    score=0.0,
    passed=False,
    critique="Reviewer response could not be parsed as valid JSON.",
    actionable_revisions=["Retry: ensure the reviewer responds with JSON only."],
)


def format_critique(feedback: ReviewFeedback) -> str:
    critique_text = feedback.critique
    if feedback.actionable_revisions:
        bullets = "\n".join(f"- {r}" for r in feedback.actionable_revisions)
        critique_text = f"{critique_text}\n\nActionable revisions:\n{bullets}"
    return critique_text
