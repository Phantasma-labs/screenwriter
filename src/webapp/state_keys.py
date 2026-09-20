# src/webapp/state_keys.py
from __future__ import annotations

THREAD_ID = "thread_id"
STAGE = "stage"
PENDING_QUESTION = "pending_question"
FINAL_RESULT = "final_result"
ERROR_MESSAGE = "error_message"
ENABLE_SEARCH = "enable_search"

STAGE_SETUP = "setup"
STAGE_RUNNING = "running"
STAGE_AWAITING_ANSWER = "awaiting_answer"
STAGE_DONE = "done"
STAGE_ERROR = "error"
