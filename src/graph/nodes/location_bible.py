# src/graph/nodes/location_bible.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.bible import parse_location_entries, render_location_bible
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills.base import T2I_PROMPT_GUIDELINES

_SYSTEM_PROMPT = (
    "You identify every key location in a script and write one production "
    "bible entry per location.\n\n" + T2I_PROMPT_GUIDELINES
)

_RESPONSE_INSTRUCTION = (
    'Respond with a JSON array only: [{"name": str, "description": str, "mood": str, '
    '"t2i_prompt": str}, ...]'
)


def make_location_bible_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def location_bible_node(state: ScreenplayState) -> NodeUpdate:
        feedback_note = (
            f"\n\nAddress this reviewer feedback:\n{state['bible_review_feedback']}"
            if state["bible_review_feedback"]
            else ""
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"{_RESPONSE_INSTRUCTION}{feedback_note}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entries = parse_location_entries(str(response.content))
        return {"location_bible": render_location_bible(entries)}

    return location_bible_node
