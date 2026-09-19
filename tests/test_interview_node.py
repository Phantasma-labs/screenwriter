# tests/test_interview_node.py
from __future__ import annotations

from unittest.mock import patch

from src.graph.nodes.interviewer import interview_wait_node, make_interview_ask_node
from src.graph.state import new_initial_state
from tests.conftest import FakeChatModel


def _state(**overrides: object):
    state = new_initial_state(
        topic="A heist film",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_interview_ask_node_autonomous_skips_llm_call():
    llm = FakeChatModel(responses=["should never be used"])
    node = make_interview_ask_node(llm=llm)
    result = node(_state(autonomous=True))
    assert result == {"interview_complete": True, "pending_question": ""}
    assert llm._call_count == 0


def test_interview_ask_node_asks_question_when_more_info_needed():
    llm = FakeChatModel(responses=['{"has_enough_info": false, "question": "What genre?"}'])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"pending_question": "What genre?"}


def test_interview_ask_node_completes_when_llm_has_enough_info():
    llm = FakeChatModel(responses=['{"has_enough_info": true, "question": ""}'])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"interview_complete": True, "pending_question": ""}


def test_interview_ask_node_completes_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"interview_complete": True, "pending_question": ""}


def test_interview_wait_node_records_turn_from_resume_value():
    state = _state()
    state["pending_question"] = "What genre?"
    with patch(
        "src.graph.nodes.interviewer.interrupt",
        return_value={"answer": "Noir thriller", "autonomous": False},
    ):
        result = interview_wait_node(state)
    expected_turn = {"question": "What genre?", "answer": "Noir thriller"}
    assert result["interview_transcript"] == [expected_turn]
    assert result["interview_turn_count"] == 1
    assert result["autonomous"] is False
    assert result["pending_question"] == ""


def test_interview_wait_node_propagates_autonomous_from_resume():
    state = _state()
    state["pending_question"] = "What genre?"
    with patch(
        "src.graph.nodes.interviewer.interrupt",
        return_value={"answer": "/finish", "autonomous": True},
    ):
        result = interview_wait_node(state)
    assert result["autonomous"] is True
