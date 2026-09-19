# src/graph/nodes/interviewer.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from src.config import get_llm
from src.graph.state import InterviewTurn, NodeUpdate, ScreenplayState
from src.utils import extract_json_object


class InterviewDecision(BaseModel):
    has_enough_info: bool
    question: str = ""


_SYSTEM_PROMPT = (
    "You are a screenwriting collaborator interviewing the user to sharpen "
    "creative direction before drafting. Ask exactly one focused question at "
    "a time about tone, character, structure, or constraints not yet covered "
    "by the topic, uploaded context, or prior answers. "
    'Respond with JSON only: {"has_enough_info": bool, "question": str}. '
    'Set has_enough_info to true (and question to "") once you have enough '
    "direction to write a strong first draft."
)


def make_interview_ask_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.5)

    def interview_ask_node(state: ScreenplayState) -> NodeUpdate:
        if state["autonomous"]:
            return {"interview_complete": True, "pending_question": ""}

        transcript_text = (
            "\n".join(
                f"Q: {turn['question']}\nA: {turn['answer']}"
                for turn in state["interview_transcript"]
            )
            or "(none yet)"
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\nSkill: {state['skill']}\n"
                    f"Context summary: {state['parsed_context'][:2000]}\n\n"
                    f"Interview so far:\n{transcript_text}"
                )
            ),
        ]
        response = llm.invoke(messages)
        decision = extract_json_object(str(response.content), InterviewDecision)
        if decision is None or decision.has_enough_info or not decision.question:
            return {"interview_complete": True, "pending_question": ""}
        return {"pending_question": decision.question}

    return interview_ask_node


def interview_wait_node(state: ScreenplayState) -> NodeUpdate:
    resumed = interrupt(state["pending_question"])
    turn: InterviewTurn = {"question": state["pending_question"], "answer": resumed["answer"]}
    return {
        "interview_transcript": state["interview_transcript"] + [turn],
        "interview_turn_count": state["interview_turn_count"] + 1,
        "autonomous": bool(resumed.get("autonomous", False)),
        "pending_question": "",
    }
