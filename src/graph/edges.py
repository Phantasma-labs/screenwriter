# src/graph/edges.py
from __future__ import annotations

from src.config import Settings, load_settings
from src.graph.state import ScreenplayState


def route_after_interview_ask(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if state["interview_complete"]:
        return "researcher"
    if state["autonomous"]:
        return "researcher"
    if state["interview_turn_count"] >= settings.max_interview_questions:
        return "researcher"
    return "interview_wait"


def route_after_interview_wait(state: ScreenplayState) -> str:
    if state["autonomous"]:
        return "researcher"
    return "interview_ask"


def route_after_reviewer(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if (
        state["review_score"] >= settings.review_pass_score
        or state["revision_count"] >= state["max_revisions"]
    ):
        return "overview"
    return "writer"


def route_after_overview(state: ScreenplayState) -> str:
    if state["autonomous"]:
        return "character_bible"
    return "overview_discussion_wait"


def route_after_overview_discussion_wait(state: ScreenplayState) -> str:
    if state["overview_discussion_finished"]:
        return "character_bible"
    return "overview_revise"


def route_after_bible_reviewer(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if (
        state["bible_review_score"] >= settings.review_pass_score
        or state["bible_revision_count"] >= state["max_bible_revisions"]
    ):
        return "finalize"
    return "character_bible"
