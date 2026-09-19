# src/graph/nodes/reviewer.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.nodes.review_shared import (
    FALLBACK_REVIEW_FEEDBACK,
    REVIEW_RESPONSE_INSTRUCTION,
    ReviewFeedback,
    format_critique,
)
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills import get_skill
from src.utils import extract_json_object


def make_reviewer_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.0)

    def reviewer_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        messages = [
            SystemMessage(content=skill.review_system_prompt()),
            HumanMessage(
                content=f"Script to review:\n\n{state['draft']}\n\n{REVIEW_RESPONSE_INSTRUCTION}"
            ),
        ]
        response = llm.invoke(messages)
        feedback = (
            extract_json_object(str(response.content), ReviewFeedback) or FALLBACK_REVIEW_FEEDBACK
        )
        return {
            "review_score": feedback.score,
            "review_feedback": format_critique(feedback),
            "status": "reviewed",
        }

    return reviewer_node
