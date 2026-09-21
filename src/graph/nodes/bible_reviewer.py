# src/graph/nodes/bible_reviewer.py
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
from src.utils import extract_json_object

_SYSTEM_PROMPT = (
    "You are a production bible editor. Review the character and location bibles "
    "together against the screenplay draft they were derived from. Check: every "
    "principal character from the draft is represented, descriptions are consistent "
    "with how the character/location reads in the script, every entry has all "
    "required fields filled in with concrete (not generic) detail, and every T2I "
    "prompt (headshot, contact sheet, wardrobe, location) is a full ~150-250 word "
    "cinematic paragraph covering subject, action, setting, composition, "
    "camera/lens, and lighting - not a short sentence."
)


def make_bible_reviewer_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.0)

    def bible_reviewer_node(state: ScreenplayState) -> NodeUpdate:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"Character Bible:\n\n{state['character_bible']}\n\n"
                    f"Location Bible:\n\n{state['location_bible']}\n\n{REVIEW_RESPONSE_INSTRUCTION}"
                )
            ),
        ]
        response = llm.invoke(messages)
        feedback = (
            extract_json_object(str(response.content), ReviewFeedback) or FALLBACK_REVIEW_FEEDBACK
        )
        return {
            "bible_review_score": feedback.score,
            "bible_review_feedback": format_critique(feedback),
        }

    return bible_reviewer_node
