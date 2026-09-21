# src/webapp/driver.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.graph.state import ScreenplayState
from src.webapp.state_keys import STAGE_AWAITING_ANSWER, STAGE_AWAITING_OVERVIEW_FEEDBACK


@dataclass
class StreamOutcome:
    interrupted: bool
    question: str | dict[str, str] | None = None
    node_events: list[str] = field(default_factory=list)
    final_values: ScreenplayState | None = None


def run_single_pass(app: Any, current_input: Any, config: dict[str, Any]) -> StreamOutcome:
    node_events: list[str] = []
    for chunk in app.stream(current_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"]
            if not interrupt_payload:
                continue
            return StreamOutcome(
                interrupted=True,
                question=interrupt_payload[0].value,
                node_events=node_events,
            )
        node_events.extend(chunk)

    final_values = app.get_state(config).values
    return StreamOutcome(interrupted=False, node_events=node_events, final_values=final_values)


def stage_for_interrupt(question: object) -> str:
    if isinstance(question, dict) and question.get("kind") == "overview_discussion":
        return STAGE_AWAITING_OVERVIEW_FEEDBACK
    return STAGE_AWAITING_ANSWER
