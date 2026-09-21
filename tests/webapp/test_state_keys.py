# tests/webapp/test_state_keys.py
from __future__ import annotations

from src.webapp import state_keys


def test_session_state_keys_are_unique():
    keys = [
        state_keys.THREAD_ID,
        state_keys.STAGE,
        state_keys.PENDING_QUESTION,
        state_keys.FINAL_RESULT,
        state_keys.ERROR_MESSAGE,
        state_keys.ENABLE_SEARCH,
    ]
    assert len(keys) == len(set(keys))


def test_stage_constants_are_unique():
    stages = [
        state_keys.STAGE_SETUP,
        state_keys.STAGE_RUNNING,
        state_keys.STAGE_AWAITING_ANSWER,
        state_keys.STAGE_AWAITING_OVERVIEW_FEEDBACK,
        state_keys.STAGE_DONE,
        state_keys.STAGE_ERROR,
    ]
    assert len(stages) == len(set(stages))
