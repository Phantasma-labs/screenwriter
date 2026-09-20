# src/graph/nodes/overview.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.overview import FALLBACK_OVERVIEW, parse_overview, render_overview
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills import get_skill

_SYSTEM_PROMPT = (
    "You write a single production overview for a finished screenplay: a short "
    "story description plus realistic technical specs for how it would actually "
    "be shot, consistent with its skill/format and tone."
)

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only: {"story_description": str, "duration_estimate": str, '
    '"frame_format": str, "aspect_ratio": str, "camera": str, "lenses": str}'
)


def make_overview_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def overview_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Skill: {skill.display_name}\n\n"
                    f"Screenplay draft:\n\n{state['draft']}\n\n{_RESPONSE_INSTRUCTION}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entry = parse_overview(str(response.content)) or FALLBACK_OVERVIEW
        return {"overview": render_overview(entry)}

    return overview_node
