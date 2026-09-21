# src/graph/nodes/overview_discussion.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.config import get_llm
from src.formatters.overview import parse_overview, render_overview
from src.graph.state import NodeUpdate, OverviewDiscussionTurn, ScreenplayState
from src.skills import get_skill

_SYSTEM_PROMPT = (
    "You revise a production overview for a screenplay based on the user's "
    "feedback. Keep everything from the current overview that the user did "
    "not ask to change, and apply their feedback precisely."
)

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only: {"story_description": str, "duration_estimate": str, '
    '"frame_format": str, "aspect_ratio": str, "camera": str, "lenses": str}'
)


def overview_discussion_wait_node(state: ScreenplayState) -> NodeUpdate:
    resumed = interrupt({"kind": "overview_discussion", "overview": state["overview"]})
    feedback = resumed.get("feedback", "")
    finish = bool(resumed.get("finish", False))
    transcript = state["overview_discussion_transcript"]
    if feedback:
        turn: OverviewDiscussionTurn = {
            "feedback": feedback,
            "overview_snapshot": state["overview"],
        }
        transcript = transcript + [turn]
    return {
        "overview_discussion_transcript": transcript,
        "overview_discussion_finished": finish,
    }


def make_overview_revise_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def overview_revise_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        latest_feedback = state["overview_discussion_transcript"][-1]["feedback"]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Skill: {skill.display_name}\n\n"
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"Current overview:\n\n{state['overview']}\n\n"
                    f"User feedback:\n\n{latest_feedback}\n\n{_RESPONSE_INSTRUCTION}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entry = parse_overview(str(response.content))
        if entry is None:
            return {"overview": state["overview"]}
        return {"overview": render_overview(entry)}

    return overview_revise_node
