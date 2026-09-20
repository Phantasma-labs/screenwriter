# tests/test_graph_edges.py
from __future__ import annotations

from src.config import Settings
from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_reviewer,
)
from src.graph.state import new_initial_state


def _settings(**overrides: object) -> Settings:
    base = dict(
        ollama_model_name="m",
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        ollama_embed_model="e",
        ollama_embed_base_url="http://localhost:11434",
        tavily_api_key=None,
        rag_chunk_size=1000,
        rag_chunk_overlap=150,
        rag_top_k=5,
        rag_min_chars_to_index=4000,
        max_interview_questions=3,
        review_pass_score=8.0,
        max_bible_revisions=1,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _state(**overrides: object):
    state = new_initial_state(
        topic="t", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_route_after_interview_ask_continues_when_more_questions_needed():
    state = _state(interview_complete=False, autonomous=False, interview_turn_count=1)
    assert route_after_interview_ask(state, _settings()) == "interview_wait"


def test_route_after_interview_ask_proceeds_when_complete():
    assert route_after_interview_ask(_state(interview_complete=True), _settings()) == "researcher"


def test_route_after_interview_ask_proceeds_when_autonomous():
    state = _state(interview_complete=False, autonomous=True)
    assert route_after_interview_ask(state, _settings()) == "researcher"


def test_route_after_interview_ask_proceeds_at_turn_cap():
    state = _state(interview_complete=False, autonomous=False, interview_turn_count=3)
    assert route_after_interview_ask(state, _settings(max_interview_questions=3)) == "researcher"


def test_route_after_interview_wait_loops_back_by_default():
    assert route_after_interview_wait(_state(autonomous=False)) == "interview_ask"


def test_route_after_interview_wait_exits_when_autonomous():
    assert route_after_interview_wait(_state(autonomous=True)) == "researcher"


def test_route_after_reviewer_passes_on_high_score():
    state = _state(review_score=8.5, revision_count=0, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "overview"


def test_route_after_reviewer_revises_on_low_score_under_cap():
    state = _state(review_score=4.0, revision_count=0, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "writer"


def test_route_after_reviewer_halts_at_revision_cap():
    state = _state(review_score=4.0, revision_count=2, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "overview"


def test_route_after_bible_reviewer_passes_on_high_score():
    state = _state(bible_review_score=9.0, bible_revision_count=0, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "finalize"


def test_route_after_bible_reviewer_revises_under_cap():
    state = _state(bible_review_score=3.0, bible_revision_count=0, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "character_bible"


def test_route_after_bible_reviewer_halts_at_cap():
    state = _state(bible_review_score=3.0, bible_revision_count=1, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "finalize"
