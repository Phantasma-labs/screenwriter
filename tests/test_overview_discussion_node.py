# tests/test_overview_discussion_node.py
from __future__ import annotations

from unittest.mock import patch

from src.graph.nodes.overview_discussion import (
    make_overview_revise_node,
    overview_discussion_wait_node,
)
from src.graph.state import new_initial_state
from tests.conftest import FakeChatModel


def _state(**overrides: object):
    state = new_initial_state(
        topic="A retired detective solves crimes via voicemail",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_overview_discussion_wait_records_feedback_and_continues():
    state = _state(overview="# Overview\n\nOriginal text.\n")
    with patch(
        "src.graph.nodes.overview_discussion.interrupt",
        return_value={"feedback": "Make the tone darker.", "finish": False},
    ):
        result = overview_discussion_wait_node(state)
    assert result["overview_discussion_transcript"] == [
        {"feedback": "Make the tone darker.", "overview_snapshot": "# Overview\n\nOriginal text.\n"}
    ]
    assert result["overview_discussion_finished"] is False
    assert result["overview_discussion_has_feedback"] is True


def test_overview_discussion_wait_skips_transcript_entry_on_bare_finish():
    state = _state(overview="# Overview\n\nOriginal text.\n")
    with patch(
        "src.graph.nodes.overview_discussion.interrupt",
        return_value={"feedback": "", "finish": True},
    ):
        result = overview_discussion_wait_node(state)
    assert result["overview_discussion_transcript"] == []
    assert result["overview_discussion_finished"] is True
    assert result["overview_discussion_has_feedback"] is False


def test_overview_discussion_wait_passes_current_overview_to_interrupt():
    state = _state(overview="# Overview\n\nCurrent draft.\n")
    with patch(
        "src.graph.nodes.overview_discussion.interrupt",
        return_value={"feedback": "", "finish": True},
    ) as mock_interrupt:
        overview_discussion_wait_node(state)
    mock_interrupt.assert_called_once_with(
        {"kind": "overview_discussion", "overview": "# Overview\n\nCurrent draft.\n"}
    )


_REVISED_OVERVIEW_JSON = (
    '{"story_description": "A retired detective solves crimes via voicemail, now darker.", '
    '"duration_estimate": "8-10 minutes", "frame_format": "Digital Cinema, 4K", '
    '"aspect_ratio": "2.39:1", "camera": "ARRI Alexa Mini", "lenses": "35mm prime"}'
)


def test_overview_revise_node_regenerates_overview_from_feedback():
    llm = FakeChatModel(responses=[_REVISED_OVERVIEW_JSON])
    node = make_overview_revise_node(llm=llm)
    state = _state(
        draft="INT. OFFICE - DAY\n\nJANE stares at the phone.",
        overview="# Overview\n\nOriginal text.\n",
        overview_discussion_transcript=[
            {
                "feedback": "Make the tone darker.",
                "overview_snapshot": "# Overview\n\nOriginal text.\n",
            }
        ],
    )
    result = node(state)
    assert "now darker" in result["overview"]


def test_overview_revise_node_keeps_prior_overview_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_overview_revise_node(llm=llm)
    state = _state(
        draft="draft text",
        overview="# Overview\n\nOriginal text.\n",
        overview_discussion_transcript=[
            {"feedback": "Make it shorter.", "overview_snapshot": "# Overview\n\nOriginal text.\n"}
        ],
    )
    result = node(state)
    assert result["overview"] == "# Overview\n\nOriginal text.\n"


def test_overview_revise_node_returns_unchanged_overview_when_transcript_empty():
    llm = FakeChatModel(responses=["should never be called"])
    node = make_overview_revise_node(llm=llm)
    state = _state(overview="# Overview\n\nOriginal text.\n", overview_discussion_transcript=[])
    result = node(state)
    assert result["overview"] == "# Overview\n\nOriginal text.\n"
    assert llm._call_count == 0


def test_overview_revise_node_system_prompt_preserves_synopsis_depth():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=[_REVISED_OVERVIEW_JSON])
    node = make_overview_revise_node(llm=llm)
    state = _state(
        draft="draft text",
        overview="# Overview\n\nOriginal text.\n",
        overview_discussion_transcript=[
            {
                "feedback": "Make the tone darker.",
                "overview_snapshot": "# Overview\n\nOriginal text.\n",
            }
        ],
    )
    node(state)
    system_content = str(captured_messages[0][0].content).lower()
    assert "synopsis" in system_content
    assert "not a one-line logline" in system_content
