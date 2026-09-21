# src/graph/state.py
from __future__ import annotations

from typing import Any, TypedDict

from src.tools.search import SearchResult


class InterviewTurn(TypedDict):
    question: str
    answer: str


class OverviewDiscussionTurn(TypedDict):
    feedback: str
    overview_snapshot: str


class ScreenplayState(TypedDict):
    topic: str
    skill: str
    file_paths: list[str]
    parsed_context: str
    rag_run_id: str
    rag_indexed: bool
    research_notes: str
    search_results: list[SearchResult]
    interview_transcript: list[InterviewTurn]
    interview_turn_count: int
    interview_complete: bool
    pending_question: str
    autonomous: bool
    outline: str
    draft: str
    review_feedback: str
    review_score: float
    revision_count: int
    max_revisions: int
    overview: str
    overview_discussion_transcript: list[OverviewDiscussionTurn]
    overview_discussion_finished: bool
    overview_discussion_has_feedback: bool
    character_bible: str
    location_bible: str
    bible_review_feedback: str
    bible_review_score: float
    bible_revision_count: int
    max_bible_revisions: int
    fountain_script: str
    screenplay_markdown: str
    status: str


NodeUpdate = dict[str, Any]
"""Partial-state update returned by LangGraph nodes. `Any` is intentional
here (and only here) - LangGraph's functional-update pattern returns a
dict with heterogeneous value types; every other signature in this
codebase uses concrete types."""


def new_initial_state(
    *,
    topic: str,
    skill: str,
    file_paths: list[str],
    max_revisions: int,
    max_bible_revisions: int,
    autonomous: bool = False,
) -> ScreenplayState:
    return ScreenplayState(
        topic=topic,
        skill=skill,
        file_paths=file_paths,
        parsed_context="",
        rag_run_id="",
        rag_indexed=False,
        research_notes="",
        search_results=[],
        interview_transcript=[],
        interview_turn_count=0,
        interview_complete=False,
        pending_question="",
        autonomous=autonomous,
        outline="",
        draft="",
        review_feedback="",
        review_score=0.0,
        revision_count=0,
        max_revisions=max_revisions,
        overview="",
        overview_discussion_transcript=[],
        overview_discussion_finished=False,
        overview_discussion_has_feedback=False,
        character_bible="",
        location_bible="",
        bible_review_feedback="",
        bible_review_score=0.0,
        bible_revision_count=0,
        max_bible_revisions=max_bible_revisions,
        fountain_script="",
        screenplay_markdown="",
        status="initialized",
    )
