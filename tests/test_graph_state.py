# tests/test_graph_state.py
from __future__ import annotations

from src.graph.state import new_initial_state


def test_new_initial_state_has_expected_defaults():
    state = new_initial_state(
        topic="A heist gone wrong",
        skill="short_film",
        file_paths=["notes.md"],
        max_revisions=2,
        max_bible_revisions=1,
    )
    assert state["topic"] == "A heist gone wrong"
    assert state["skill"] == "short_film"
    assert state["file_paths"] == ["notes.md"]
    assert state["parsed_context"] == ""
    assert state["interview_transcript"] == []
    assert state["interview_turn_count"] == 0
    assert state["interview_complete"] is False
    assert state["pending_question"] == ""
    assert state["autonomous"] is False
    assert state["review_score"] == 0.0
    assert state["revision_count"] == 0
    assert state["max_revisions"] == 2
    assert state["max_bible_revisions"] == 1
    assert state["status"] == "initialized"


def test_new_initial_state_autonomous_flag():
    state = new_initial_state(
        topic="x",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=True,
    )
    assert state["autonomous"] is True


def test_new_initial_state_is_mutable_dict_for_langgraph_updates():
    state = new_initial_state(
        topic="x", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state["draft"] = "INT. ROOM - DAY\n\nShe waits.\n"
    assert state["draft"] == "INT. ROOM - DAY\n\nShe waits.\n"


def test_new_initial_state_has_overview_discussion_defaults():
    state = new_initial_state(
        topic="x", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    assert state["overview_discussion_transcript"] == []
    assert state["overview_discussion_finished"] is False
