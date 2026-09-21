# tests/webapp/test_driver.py
from __future__ import annotations

from src.webapp.driver import run_single_pass, stage_for_interrupt
from src.webapp.state_keys import STAGE_AWAITING_ANSWER, STAGE_AWAITING_OVERVIEW_FEEDBACK


class _FakeInterrupt:
    def __init__(self, value: str | dict[str, str]) -> None:
        self.value = value


class _FakeState:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values


class _FakeApp:
    def __init__(self, batch: list[dict[str, object]], final_values: dict[str, object]) -> None:
        self._batch = batch
        self._final_values = final_values

    def stream(self, current_input, config, stream_mode="updates"):
        yield from self._batch

    def get_state(self, config):
        return _FakeState(self._final_values)


def test_run_single_pass_returns_interrupted_outcome_with_question():
    app = _FakeApp(
        [{"ingest": {}}, {"__interrupt__": (_FakeInterrupt("What tone?"),)}],
        final_values={},
    )
    outcome = run_single_pass(app, {"topic": "t"}, {"configurable": {"thread_id": "x"}})
    assert outcome.interrupted is True
    assert outcome.question == "What tone?"
    assert outcome.node_events == ["ingest"]
    assert outcome.final_values is None


def test_run_single_pass_skips_empty_interrupt_payload_and_completes():
    app = _FakeApp(
        [
            {"ingest": {}},
            {"__interrupt__": ()},
            {"writer": {}},
            {"finalize": {}},
        ],
        final_values={"status": "finalized"},
    )
    outcome = run_single_pass(app, {"topic": "t"}, {"configurable": {"thread_id": "x"}})
    assert outcome.interrupted is False
    assert outcome.node_events == ["ingest", "writer", "finalize"]
    assert outcome.final_values == {"status": "finalized"}


def test_run_single_pass_completes_immediately_when_no_interrupt():
    app = _FakeApp(
        [{"ingest": {}}, {"outliner": {}}, {"writer": {}}, {"reviewer": {}}, {"finalize": {}}],
        final_values={"status": "finalized"},
    )
    outcome = run_single_pass(app, {"topic": "t"}, {"configurable": {"thread_id": "x"}})
    assert outcome.interrupted is False
    assert outcome.node_events == ["ingest", "outliner", "writer", "reviewer", "finalize"]
    assert outcome.final_values == {"status": "finalized"}


def test_stage_for_interrupt_returns_overview_feedback_stage_for_overview_discussion():
    question = {"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}
    assert stage_for_interrupt(question) == STAGE_AWAITING_OVERVIEW_FEEDBACK


def test_stage_for_interrupt_returns_answer_stage_for_interview_question():
    assert stage_for_interrupt("What tone should this have?") == STAGE_AWAITING_ANSWER


def test_run_single_pass_returns_interrupted_outcome_with_dict_question():
    question_payload = {
        "kind": "overview_discussion",
        "overview": "# Overview\n\nDraft.",
    }
    app = _FakeApp(
        [
            {"overview": {}},
            {"__interrupt__": (_FakeInterrupt(question_payload),)},
        ],
        final_values={},
    )
    outcome = run_single_pass(app, {"topic": "t"}, {"configurable": {"thread_id": "x"}})
    assert outcome.interrupted is True
    assert outcome.question == question_payload
