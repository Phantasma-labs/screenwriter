# app.py
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from main import STATUS_BADGES, SUPPORTED_CONTEXT_EXTENSIONS, is_finish_command
from src.config import ConfigError, load_settings
from src.graph.state import new_initial_state
from src.graph.workflow import build_workflow
from src.output_files import build_output_files
from src.parsers.base import ParserError
from src.skills import available_skills
from src.webapp.archive import build_markdown_zip
from src.webapp.driver import run_single_pass, stage_for_interrupt
from src.webapp.state_keys import (
    ENABLE_SEARCH,
    ERROR_MESSAGE,
    FINAL_RESULT,
    PENDING_QUESTION,
    STAGE,
    STAGE_AWAITING_ANSWER,
    STAGE_AWAITING_OVERVIEW_FEEDBACK,
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_SETUP,
    THREAD_ID,
)
from src.webapp.uploads import save_uploaded_files

st.set_page_config(page_title="Screenwriter Agent", layout="wide")


@st.cache_resource
def get_compiled_app(enable_search: bool) -> Any:
    return build_workflow(enable_search=enable_search)


def _init_session_state() -> None:
    if THREAD_ID not in st.session_state:
        st.session_state[THREAD_ID] = str(uuid.uuid4())
    if STAGE not in st.session_state:
        st.session_state[STAGE] = STAGE_SETUP


def _reset_session_state() -> None:
    st.session_state[THREAD_ID] = str(uuid.uuid4())
    st.session_state[STAGE] = STAGE_SETUP
    st.session_state.pop(PENDING_QUESTION, None)
    st.session_state.pop(FINAL_RESULT, None)
    st.session_state.pop(ERROR_MESSAGE, None)
    st.session_state.pop(ENABLE_SEARCH, None)


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state[THREAD_ID]}}


def _drive(app: Any, current_input: Any, config: dict[str, Any]) -> None:
    try:
        with st.status("Working...", expanded=True) as status_box:
            outcome = run_single_pass(app, current_input, config)
            for node_name in outcome.node_events:
                status_box.write(STATUS_BADGES.get(node_name, f"[{node_name.upper()}]"))
        if outcome.interrupted:
            st.session_state[PENDING_QUESTION] = outcome.question
            st.session_state[STAGE] = stage_for_interrupt(outcome.question)
        else:
            st.session_state[FINAL_RESULT] = outcome.final_values
            st.session_state[STAGE] = STAGE_DONE
    except (ConfigError, ParserError) as exc:
        st.session_state[ERROR_MESSAGE] = str(exc)
        st.session_state[STAGE] = STAGE_ERROR
    except Exception as exc:  # last-resort UI guard
        st.session_state[ERROR_MESSAGE] = f"Unexpected error: {exc}"
        st.session_state[STAGE] = STAGE_ERROR


def _render_setup(settings: Any) -> None:
    st.title("Screenwriter Agent")
    topic = st.text_area("Topic / premise", height=120)
    skill = st.selectbox("Skill", available_skills())
    uploaded_files = st.file_uploader(
        "Context files",
        accept_multiple_files=True,
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_CONTEXT_EXTENSIONS),
    )
    enable_search = st.checkbox("Enable web research")
    autonomous = st.checkbox("Autonomous (skip the interview)")
    max_revisions = st.number_input("Max revisions", min_value=0, max_value=10, value=2)

    if st.button("Start", type="primary", disabled=not topic.strip()):
        st.session_state[ENABLE_SEARCH] = enable_search
        app = get_compiled_app(enable_search)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_paths = save_uploaded_files(uploaded_files or [], Path(tmp_dir))
            initial_state = new_initial_state(
                topic=topic,
                skill=skill,
                file_paths=file_paths,
                max_revisions=int(max_revisions),
                max_bible_revisions=settings.max_bible_revisions,
                autonomous=autonomous,
            )
            _drive(app, initial_state, _config())
        st.rerun()


def _render_awaiting_answer() -> None:
    app = get_compiled_app(st.session_state.get(ENABLE_SEARCH, False))
    st.title("Screenwriter Agent")
    transcript = app.get_state(_config()).values.get("interview_transcript", [])
    for turn in transcript:
        st.chat_message("assistant").write(turn["question"])
        st.chat_message("user").write(turn["answer"])
    st.chat_message("assistant").write(st.session_state[PENDING_QUESTION])
    answer = st.text_input("Your answer")
    col1, col2 = st.columns(2)
    if col1.button("Submit answer", type="primary"):
        resume = Command(resume={"answer": answer, "autonomous": is_finish_command(answer)})
        _drive(app, resume, _config())
        st.rerun()
    if col2.button("Finish now, write autonomously"):
        resume = Command(resume={"answer": "", "autonomous": True})
        _drive(app, resume, _config())
        st.rerun()


def _render_awaiting_overview_feedback() -> None:
    app = get_compiled_app(st.session_state.get(ENABLE_SEARCH, False))
    st.title("Screenwriter Agent")
    payload = st.session_state[PENDING_QUESTION]
    st.subheader("Overview pre-result")
    st.markdown(payload["overview"])
    transcript = app.get_state(_config()).values.get("overview_discussion_transcript", [])
    for turn in transcript:
        st.chat_message("user").write(turn["feedback"])
    feedback = st.text_area("Feedback (leave blank and accept to continue)")
    col1, col2 = st.columns(2)
    if col1.button("Revise", type="primary", disabled=not feedback.strip()):
        resume = Command(resume={"feedback": feedback, "finish": False})
        _drive(app, resume, _config())
        st.rerun()
    if col2.button("Accept and continue"):
        resume = Command(resume={"feedback": "", "finish": True})
        _drive(app, resume, _config())
        st.rerun()


def _render_done() -> None:
    col_title, col_restart = st.columns([5, 1])
    with col_title:
        st.title("Screenplay ready")
    with col_restart:
        if st.button("Restart"):
            _reset_session_state()
            st.rerun()

    result = st.session_state[FINAL_RESULT]
    outputs = build_output_files(result)

    tab_overview, tab_script, tab_md, tab_chars, tab_locs, tab_refs = st.tabs(
        ["Overview", "Script", "Screenplay", "Character Bible", "Location Bible", "References"]
    )
    with tab_overview:
        st.markdown(outputs["Overview.md"])
        st.download_button(
            "Download MDs",
            build_markdown_zip(outputs),
            file_name="Screenplay.zip",
            mime="application/zip",
        )
    with tab_script:
        st.code(outputs["Script.md"], language="text")
    with tab_md:
        st.markdown(outputs["Screenplay.md"])
    with tab_chars:
        st.markdown(outputs["CharacterBible.md"])
    with tab_locs:
        st.markdown(outputs["LocationBible.md"])

    with tab_refs:
        references = result.get("search_results", [])
        if not references:
            st.info("No web references were found for this run.")
        else:
            for ref in references:
                st.markdown(f"- [{ref['title']}]({ref['url']}) — {ref['snippet']}")


def _render_error() -> None:
    st.title("Screenwriter Agent")
    st.error(st.session_state.get(ERROR_MESSAGE, "An unknown error occurred."))
    if st.button("Back to setup"):
        _reset_session_state()
        st.rerun()


def main() -> None:
    try:
        settings = load_settings()
        get_compiled_app(False)
    except ConfigError as exc:
        st.title("Screenwriter Agent")
        st.error(f"Configuration error: {exc}")
        st.stop()

    _init_session_state()
    stage = st.session_state[STAGE]

    if stage == STAGE_SETUP:
        _render_setup(settings)
    elif stage == STAGE_AWAITING_ANSWER:
        _render_awaiting_answer()
    elif stage == STAGE_AWAITING_OVERVIEW_FEEDBACK:
        _render_awaiting_overview_feedback()
    elif stage == STAGE_DONE:
        _render_done()
    else:
        _render_error()


main()
