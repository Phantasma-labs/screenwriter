# Overview Discussion Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a human-in-the-loop discussion stage between the `overview` node and `character_bible` node, so the LLM's first overview is treated as a pre-result the user can iteratively critique before the pipeline generates the bibles, script, and screenplay.

**Architecture:** Two new LangGraph nodes — `overview_discussion_wait` (a pure `interrupt()` boundary, mirroring `interview_wait`) and `overview_revise` (an LLM call that regenerates the overview against the latest feedback, mirroring `overview`) — are wired into a loop between `overview` and `character_bible`. `--autonomous` runs bypass the loop entirely via a new conditional edge. The CLI and webapp both already have a generic `interrupt()`/`Command(resume=...)` drive loop from the interview stage; this plan extends both to recognize a second, dict-shaped interrupt payload (`{"kind": "overview_discussion", ...}`) alongside the interview's plain-string payload.

**Tech Stack:** Python 3.10+, LangGraph (`StateGraph`, `interrupt`/`Command`), Pydantic v2, pytest, `FakeChatModel` test double (`tests/conftest.py`), Streamlit (`app.py`).

**Spec:** [docs/superpowers/specs/2026-09-20-overview-discussion-loop-design.md](../specs/2026-09-20-overview-discussion-loop-design.md)

## Global Constraints

- Every file starts with `from __future__ import annotations`; imports grouped stdlib / third-party / internal (ruff `I` rule).
- Ruff line length 100, `select = ["E", "F", "I", "UP"]`. Run `ruff check --fix` and `ruff format` before each commit.
- Strict type annotations everywhere; `TypedDict` for `ScreenplayState` fields, Pydantic v2 `BaseModel` for structured LLM outputs. No bare `Any`/unparameterized `dict`/`list` outside `NodeUpdate`.
- LangGraph nodes return partial-update dicts (`NodeUpdate`), never mutate state in place. A node only writes the state key(s) it owns.
- All tests run 100% offline — no live Ollama/Tavily/DuckDuckGo/Chroma calls. LLM calls are doubled with `FakeChatModel` from `tests/conftest.py`.
- No turn cap on the overview discussion loop (explicit design decision — the human decides when to stop, unlike the capped writer/reviewer and bible loops).
- `--autonomous` bypasses the discussion loop entirely, consistent with how it already bypasses the interview.
- The interview stage's existing interrupt payload contract (a bare string) must not change.

---

### Task 1: State fields for the discussion transcript

**Files:**
- Modify: `src/graph/state.py`
- Test: `tests/test_graph_state.py`

**Interfaces:**
- Produces: `OverviewDiscussionTurn` TypedDict (`feedback: str`, `overview_snapshot: str`); `ScreenplayState` fields `overview_discussion_transcript: list[OverviewDiscussionTurn]`, `overview_discussion_finished: bool`; `new_initial_state(...)` now returns those two fields initialized to `[]` and `False`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph_state.py`:

```python
def test_new_initial_state_has_overview_discussion_defaults():
    state = new_initial_state(
        topic="x", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    assert state["overview_discussion_transcript"] == []
    assert state["overview_discussion_finished"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_state.py::test_new_initial_state_has_overview_discussion_defaults -v`
Expected: FAIL with `KeyError: 'overview_discussion_transcript'`

- [ ] **Step 3: Write minimal implementation**

In `src/graph/state.py`, add the new TypedDict right after `InterviewTurn`:

```python
class OverviewDiscussionTurn(TypedDict):
    feedback: str
    overview_snapshot: str
```

Add these two fields to `ScreenplayState`, right after `overview: str`:

```python
    overview: str
    overview_discussion_transcript: list[OverviewDiscussionTurn]
    overview_discussion_finished: bool
    character_bible: str
```

Add the matching defaults to `new_initial_state`, right after `overview=""`:

```python
        overview="",
        overview_discussion_transcript=[],
        overview_discussion_finished=False,
        character_bible="",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_state.py -v`
Expected: All PASS (including the new test and the three pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/graph/state.py tests/test_graph_state.py
git commit -m "feat: add overview discussion state fields"
```

---

### Task 2: Routing functions for the discussion loop

**Files:**
- Modify: `src/graph/edges.py`
- Test: `tests/test_graph_edges.py`

**Interfaces:**
- Consumes: `ScreenplayState` fields from Task 1 (`overview_discussion_finished`) and the pre-existing `autonomous`.
- Produces: `route_after_overview(state: ScreenplayState) -> str` (returns `"character_bible"` or `"overview_discussion_wait"`); `route_after_overview_discussion_wait(state: ScreenplayState) -> str` (returns `"character_bible"` or `"overview_revise"`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_edges.py`:

```python
from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_overview,
    route_after_overview_discussion_wait,
    route_after_reviewer,
)


def test_route_after_overview_goes_to_discussion_by_default():
    assert route_after_overview(_state(autonomous=False)) == "overview_discussion_wait"


def test_route_after_overview_skips_discussion_when_autonomous():
    assert route_after_overview(_state(autonomous=True)) == "character_bible"


def test_route_after_overview_discussion_wait_loops_back_by_default():
    state = _state(overview_discussion_finished=False)
    assert route_after_overview_discussion_wait(state) == "overview_revise"


def test_route_after_overview_discussion_wait_exits_when_finished():
    state = _state(overview_discussion_finished=True)
    assert route_after_overview_discussion_wait(state) == "character_bible"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_edges.py -v`
Expected: FAIL with `ImportError: cannot import name 'route_after_overview'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/graph/edges.py`, after `route_after_reviewer` and before `route_after_bible_reviewer`:

```python
def route_after_overview(state: ScreenplayState) -> str:
    if state["autonomous"]:
        return "character_bible"
    return "overview_discussion_wait"


def route_after_overview_discussion_wait(state: ScreenplayState) -> str:
    if state["overview_discussion_finished"]:
        return "character_bible"
    return "overview_revise"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_edges.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph/edges.py tests/test_graph_edges.py
git commit -m "feat: add routing for overview discussion loop"
```

---

### Task 3: `overview_discussion_wait` and `overview_revise` nodes

**Files:**
- Create: `src/graph/nodes/overview_discussion.py`
- Test: `tests/test_overview_discussion_node.py`

**Interfaces:**
- Consumes: `ScreenplayState`, `NodeUpdate` (`src/graph/state.py`); `OverviewDiscussionTurn` (Task 1); `parse_overview`, `render_overview` (`src/formatters/overview.py` — note: unlike `overview_node`, this node does NOT fall back to `FALLBACK_OVERVIEW` on a parse failure; it keeps the prior `state["overview"]` instead, since discarding a good prior version would be worse here); `get_llm` (`src/config.py`); `get_skill` (`src/skills`); `langgraph.types.interrupt`.
- Produces: `overview_discussion_wait_node(state: ScreenplayState) -> NodeUpdate`; `make_overview_revise_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overview_discussion_node.py`:

```python
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


def test_overview_discussion_wait_skips_transcript_entry_on_bare_finish():
    state = _state(overview="# Overview\n\nOriginal text.\n")
    with patch(
        "src.graph.nodes.overview_discussion.interrupt",
        return_value={"feedback": "", "finish": True},
    ):
        result = overview_discussion_wait_node(state)
    assert result["overview_discussion_transcript"] == []
    assert result["overview_discussion_finished"] is True


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
            {"feedback": "Make the tone darker.", "overview_snapshot": "# Overview\n\nOriginal text.\n"}
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_overview_discussion_node.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.overview_discussion'`

- [ ] **Step 3: Write minimal implementation**

Create `src/graph/nodes/overview_discussion.py`:

```python
# src/graph/nodes/overview_discussion.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.config import get_llm
from src.formatters.overview import parse_overview, render_overview
from src.graph.state import NodeUpdate, OverviewDiscussionTurn, ScreenplayState
from src.skills import get_skill

_SYSTEM_PROMPT = (
    "You revise a production overview for a screenplay based on the user's "
    "feedback. Keep everything from the current overview that the user did "
    "not ask to change, and apply their feedback precisely."
)

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only: {"story_description": str, "duration_estimate": str, '
    '"frame_format": str, "aspect_ratio": str, "camera": str, "lenses": str}'
)


def overview_discussion_wait_node(state: ScreenplayState) -> NodeUpdate:
    resumed = interrupt({"kind": "overview_discussion", "overview": state["overview"]})
    feedback = resumed.get("feedback", "")
    finish = bool(resumed.get("finish", False))
    transcript = state["overview_discussion_transcript"]
    if feedback:
        turn: OverviewDiscussionTurn = {
            "feedback": feedback,
            "overview_snapshot": state["overview"],
        }
        transcript = transcript + [turn]
    return {
        "overview_discussion_transcript": transcript,
        "overview_discussion_finished": finish,
    }


def make_overview_revise_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def overview_revise_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        latest_feedback = state["overview_discussion_transcript"][-1]["feedback"]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Skill: {skill.display_name}\n\n"
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"Current overview:\n\n{state['overview']}\n\n"
                    f"User feedback:\n\n{latest_feedback}\n\n{_RESPONSE_INSTRUCTION}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entry = parse_overview(str(response.content))
        if entry is None:
            return {"overview": state["overview"]}
        return {"overview": render_overview(entry)}

    return overview_revise_node
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_overview_discussion_node.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/overview_discussion.py tests/test_overview_discussion_node.py
git commit -m "feat: add overview discussion wait and revise nodes"
```

---

### Task 4: Wire the new nodes into the graph

**Files:**
- Modify: `src/graph/workflow.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: `overview_discussion_wait_node`, `make_overview_revise_node` (Task 3); `route_after_overview`, `route_after_overview_discussion_wait` (Task 2).
- Produces: compiled graph now contains nodes `"overview_discussion_wait"` and `"overview_revise"`, reachable from `"overview"`.

- [ ] **Step 1: Write the failing test**

Update the node-name list in `tests/test_workflow.py::test_build_workflow_compiles_with_all_expected_nodes`:

```python
def test_build_workflow_compiles_with_all_expected_nodes():
    app = build_workflow(enable_search=False)
    node_names = set(app.get_graph().nodes.keys())
    for expected in [
        "ingest",
        "interview_ask",
        "interview_wait",
        "researcher",
        "outliner",
        "writer",
        "reviewer",
        "overview",
        "overview_discussion_wait",
        "overview_revise",
        "character_bible",
        "location_bible",
        "bible_reviewer",
        "finalize",
    ]:
        assert expected in node_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py::test_build_workflow_compiles_with_all_expected_nodes -v`
Expected: FAIL — `overview_discussion_wait` not in `node_names`

- [ ] **Step 3: Write minimal implementation**

In `src/graph/workflow.py`, update the edges import:

```python
from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_overview,
    route_after_overview_discussion_wait,
    route_after_reviewer,
)
```

Add the node import:

```python
from src.graph.nodes.overview_discussion import (
    make_overview_revise_node,
    overview_discussion_wait_node,
)
```

Register the two new nodes, right after `graph.add_node("overview", make_overview_node(llm=llm))`:

```python
    graph.add_node("overview", make_overview_node(llm=llm))
    graph.add_node("overview_discussion_wait", overview_discussion_wait_node)
    graph.add_node("overview_revise", make_overview_revise_node(llm=llm))
    graph.add_node("character_bible", make_character_bible_node(llm=llm))
```

Replace the single `graph.add_edge("overview", "character_bible")` line with:

```python
    graph.add_conditional_edges(
        "overview",
        route_after_overview,
        {"overview_discussion_wait": "overview_discussion_wait", "character_bible": "character_bible"},
    )
    graph.add_conditional_edges(
        "overview_discussion_wait",
        route_after_overview_discussion_wait,
        {"overview_revise": "overview_revise", "character_bible": "character_bible"},
    )
    graph.add_edge("overview_revise", "overview_discussion_wait")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph/workflow.py tests/test_workflow.py
git commit -m "feat: wire overview discussion loop into the graph"
```

---

### Task 5: End-to-end graph behavior for the discussion loop

**Files:**
- Modify: `tests/test_workflow_e2e.py`

**Interfaces:**
- Consumes: `build_workflow` (Task 4), `FakeChatModel`, `Command` (`langgraph.types`).
- Produces: no new production code; verifies the wired graph's runtime behavior end to end.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workflow_e2e.py`:

```python
def test_full_workflow_autonomous_bypasses_overview_discussion():
    responses = [
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
        '{"story_description": "A phone rings once a year.", "duration_estimate": "5 minutes", '
        '"frame_format": "Digital, 2K", "aspect_ratio": "16:9", "camera": "Sony FX3", '
        '"lenses": "24-70mm zoom"}',
        "[]",
        "[]",
        '{"score": 9.0, "passed": true, "critique": "Fine.", "actionable_revisions": []}',
        "Create a minimalist poster of a ringing phone in a dark room.",
    ]
    llm = FakeChatModel(responses=responses)
    app = build_workflow(enable_search=False, llm=llm, search_fn=lambda q: [])

    initial_state = new_initial_state(
        topic="A phone that rings once a year",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=True,
    )
    config = {"configurable": {"thread_id": "test-thread-autonomous-overview"}}
    result = app.invoke(initial_state, config)

    assert "__interrupt__" not in result
    assert result["status"] == "finalized"
    assert result["overview_discussion_transcript"] == []


def test_workflow_overview_discussion_pauses_revises_then_finishes():
    responses = [
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
        '{"story_description": "A phone rings once a year.", "duration_estimate": "5 minutes", '
        '"frame_format": "Digital, 2K", "aspect_ratio": "16:9", "camera": "Sony FX3", '
        '"lenses": "24-70mm zoom"}',
        '{"story_description": "A phone rings once a year, now darker and quieter.", '
        '"duration_estimate": "5 minutes", "frame_format": "Digital, 2K", "aspect_ratio": "16:9", '
        '"camera": "Sony FX3", "lenses": "24-70mm zoom"}',
        "[]",
        "[]",
        '{"score": 9.0, "passed": true, "critique": "Fine.", "actionable_revisions": []}',
        "Create a minimalist poster of a ringing phone in a dark room.",
    ]
    llm = FakeChatModel(responses=responses)
    app = build_workflow(enable_search=False, llm=llm, search_fn=lambda q: [])

    initial_state = new_initial_state(
        topic="A phone that rings once a year",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=False,
    )
    config = {"configurable": {"thread_id": "test-thread-overview-discussion"}}

    paused = app.invoke(initial_state, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "overview_discussion"
    assert "A phone rings once a year." in paused["__interrupt__"][0].value["overview"]

    revised = app.invoke(
        Command(resume={"feedback": "Make it darker and quieter.", "finish": False}), config
    )
    assert "__interrupt__" in revised
    assert "now darker and quieter" in revised["__interrupt__"][0].value["overview"]

    resumed = app.invoke(Command(resume={"feedback": "", "finish": True}), config)
    assert "__interrupt__" not in resumed
    assert resumed["status"] == "finalized"
    assert resumed["overview_discussion_transcript"] == [
        {
            "feedback": "Make it darker and quieter.",
            "overview_snapshot": paused["__interrupt__"][0].value["overview"],
        }
    ]
    assert "now darker and quieter" in resumed["overview"]
```

Note: the interview stage's already-passing tests in this same file
(`test_full_workflow_autonomous_short_film_produces_all_artifacts`,
`test_workflow_interview_pauses_for_human_input_then_resumes`,
`test_workflow_bible_revise_loop_runs_twice_then_finalizes`) all run
autonomous or otherwise reach `overview` with `autonomous=True`/`False`
respectively — check each one: `test_workflow_interview_pauses_for_human_input_then_resumes`
sets `autonomous=False` on `new_initial_state` but the interview loop's
last resume sets `"autonomous": False` too, so after the interview
completes, `state["autonomous"]` stays `False` and this run will now
**also** pause at `overview_discussion_wait` — it needs one more
response in its LLM list and one more `app.invoke(Command(...))` call to
finish. Update it as part of this task (see Step 3).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow_e2e.py -v`
Expected: The two new tests FAIL (`overview_discussion_wait` doesn't exist yet
until Task 4 — if Tasks 1-4 are already done by this point, they'll fail
instead on assertion mismatches or hang on a missing interrupt); the
pre-existing `test_workflow_interview_pauses_for_human_input_then_resumes`
FAILS after this task's edit is applied, until Step 3's fix lands.

- [ ] **Step 3: Fix the pre-existing interview test and add the two new tests**

Update `test_workflow_interview_pauses_for_human_input_then_resumes` in
`tests/test_workflow_e2e.py`: it ends on `autonomous=False`, so it now
pauses again at the overview discussion stage. Change its `responses` list
and body to:

```python
def test_workflow_interview_pauses_for_human_input_then_resumes():
    responses = [
        '{"has_enough_info": false, "question": "What tone should this have?"}',
        '{"has_enough_info": true, "question": ""}',
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
        '{"story_description": "A phone rings once a year.", "duration_estimate": "5 minutes", '
        '"frame_format": "Digital, 2K", "aspect_ratio": "16:9", "camera": "Sony FX3", '
        '"lenses": "24-70mm zoom"}',
        "[]",
        "[]",
        '{"score": 9.0, "passed": true, "critique": "Fine.", "actionable_revisions": []}',
        "Create a minimalist poster of a ringing phone in a dark room.",
    ]
    llm = FakeChatModel(responses=responses)
    app = build_workflow(enable_search=False, llm=llm, search_fn=lambda q: [])

    initial_state = new_initial_state(
        topic="A phone that rings once a year",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=False,
    )
    config = {"configurable": {"thread_id": "test-thread-2"}}

    paused = app.invoke(initial_state, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == "What tone should this have?"

    paused_at_overview = app.invoke(
        Command(resume={"answer": "Melancholy and quiet.", "autonomous": False}), config
    )
    assert "__interrupt__" in paused_at_overview
    assert paused_at_overview["__interrupt__"][0].value["kind"] == "overview_discussion"

    resumed = app.invoke(Command(resume={"feedback": "", "finish": True}), config)

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "finalized"
    assert resumed["interview_transcript"] == [
        {"question": "What tone should this have?", "answer": "Melancholy and quiet."}
    ]
```

Then add the two new test functions from Step 1 anywhere else in the file
(e.g. after `test_full_workflow_autonomous_short_film_produces_all_artifacts`
and after the interview test, respectively).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow_e2e.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow_e2e.py
git commit -m "test: cover overview discussion loop end to end"
```

---

### Task 6: CLI support for the discussion loop

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `is_finish_command` (already in `main.py`); interrupt payloads shaped either as a bare string (interview) or `{"kind": "overview_discussion", "overview": str}` (this task).
- Produces: `run_interactive` now branches on payload shape; `STATUS_BADGES` gains `"overview_discussion_wait"` and `"overview_revise"` entries.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_main.py`:

```python
def test_run_interactive_handles_overview_discussion_interrupt(monkeypatch, capsys):
    batches = [
        [{"overview": {}}, {"__interrupt__": (_FakeInterrupt({"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}),)}],
        [{"character_bible": {}}, {"finalize": {}}],
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})
    monkeypatch.setattr("builtins.input", lambda prompt="": "/finish")

    result = run_interactive(
        app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}}
    )

    assert result == {"status": "finalized"}
    out = capsys.readouterr().out
    assert "# Overview" in out
    assert "Draft." in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_run_interactive_handles_overview_discussion_interrupt -v`
Expected: FAIL — the current code does `question = interrupt_payload[0].value` unconditionally and then
`print(f"[INTERVIEWING] {question}")`, treating the dict as the question, so the overview text never
appears verbatim and the resume payload sent doesn't match `{"feedback": ..., "finish": ...}`
(the fake app's second batch would still run, so this specific assertion on stdout is what catches it).

- [ ] **Step 3: Write minimal implementation**

In `main.py`, update `STATUS_BADGES`:

```python
STATUS_BADGES = {
    "ingest": "[INGESTING]",
    "interview_ask": "[INTERVIEWING]",
    "interview_wait": "[INTERVIEWING]",
    "researcher": "[RESEARCHING]",
    "outliner": "[OUTLINING]",
    "writer": "[WRITING DRAFT]",
    "reviewer": "[REVIEWING]",
    "overview": "[BUILDING OVERVIEW]",
    "overview_discussion_wait": "[OVERVIEW DISCUSSION]",
    "overview_revise": "[REVISING OVERVIEW]",
    "character_bible": "[BUILDING CHARACTER BIBLE]",
    "location_bible": "[BUILDING LOCATION BIBLE]",
    "bible_reviewer": "[REVIEWING BIBLES]",
    "finalize": "[FINALIZING]",
}
```

Replace the interrupt-handling block inside `run_interactive`:

```python
            if "__interrupt__" in chunk:
                interrupt_payload = chunk["__interrupt__"]
                if not interrupt_payload:
                    continue
                value = interrupt_payload[0].value
                if isinstance(value, dict) and value.get("kind") == "overview_discussion":
                    print("[OVERVIEW PRE-RESULT]")
                    print(value["overview"])
                    feedback = input("Feedback (or /finish to accept and continue): ")
                    finish = is_finish_command(feedback)
                    current_input = Command(
                        resume={"feedback": "" if finish else feedback, "finish": finish}
                    )
                else:
                    question = value
                    print(f"[INTERVIEWING] {question}")
                    answer = input("> ")
                    current_input = Command(
                        resume={"answer": answer, "autonomous": is_finish_command(answer)}
                    )
                interrupted = True
                break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: handle overview discussion interrupt in the CLI"
```

---

### Task 7: Webapp stage constant

**Files:**
- Modify: `src/webapp/state_keys.py`
- Test: `tests/webapp/test_state_keys.py`

**Interfaces:**
- Produces: `STAGE_AWAITING_OVERVIEW_FEEDBACK: str` in `src/webapp/state_keys.py`.

- [ ] **Step 1: Write the failing test**

Update `tests/webapp/test_state_keys.py::test_stage_constants_are_unique`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_state_keys.py -v`
Expected: FAIL with `AttributeError: module 'src.webapp.state_keys' has no attribute 'STAGE_AWAITING_OVERVIEW_FEEDBACK'`

- [ ] **Step 3: Write minimal implementation**

In `src/webapp/state_keys.py`, add after `STAGE_AWAITING_ANSWER`:

```python
STAGE_AWAITING_ANSWER = "awaiting_answer"
STAGE_AWAITING_OVERVIEW_FEEDBACK = "awaiting_overview_feedback"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_state_keys.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/webapp/state_keys.py tests/webapp/test_state_keys.py
git commit -m "feat: add webapp stage constant for overview discussion"
```

---

### Task 8: Webapp driver stage routing helper

**Files:**
- Modify: `src/webapp/driver.py`
- Test: `tests/webapp/test_driver.py`

**Interfaces:**
- Consumes: `STAGE_AWAITING_ANSWER`, `STAGE_AWAITING_OVERVIEW_FEEDBACK` (Task 7).
- Produces: `stage_for_interrupt(question: object) -> str` — a pure function so `app.py` doesn't need its own branching logic embedded in a Streamlit-context-dependent function (keeps it unit-testable without `AppTest`). `StreamOutcome.question` type widens from `str | None` to `Any | None` (it was already an opaque pass-through of the interrupt value; this task is the first to actually put a non-string value there).

- [ ] **Step 1: Write the failing tests**

Add to `tests/webapp/test_driver.py`:

```python
from src.webapp.driver import run_single_pass, stage_for_interrupt
from src.webapp.state_keys import STAGE_AWAITING_ANSWER, STAGE_AWAITING_OVERVIEW_FEEDBACK


def test_stage_for_interrupt_returns_overview_feedback_stage_for_overview_discussion():
    question = {"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}
    assert stage_for_interrupt(question) == STAGE_AWAITING_OVERVIEW_FEEDBACK


def test_stage_for_interrupt_returns_answer_stage_for_interview_question():
    assert stage_for_interrupt("What tone should this have?") == STAGE_AWAITING_ANSWER


def test_run_single_pass_returns_interrupted_outcome_with_dict_question():
    app = _FakeApp(
        [
            {"overview": {}},
            {
                "__interrupt__": (
                    _FakeInterrupt({"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}),
                )
            },
        ],
        final_values={},
    )
    outcome = run_single_pass(app, {"topic": "t"}, {"configurable": {"thread_id": "x"}})
    assert outcome.interrupted is True
    assert outcome.question == {"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/webapp/test_driver.py -v`
Expected: FAIL with `ImportError: cannot import name 'stage_for_interrupt'`

- [ ] **Step 3: Write minimal implementation**

In `src/webapp/driver.py`:

```python
# src/webapp/driver.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.graph.state import ScreenplayState
from src.webapp.state_keys import STAGE_AWAITING_ANSWER, STAGE_AWAITING_OVERVIEW_FEEDBACK


@dataclass
class StreamOutcome:
    interrupted: bool
    question: Any | None = None
    node_events: list[str] = field(default_factory=list)
    final_values: ScreenplayState | None = None


def run_single_pass(app: Any, current_input: Any, config: dict[str, Any]) -> StreamOutcome:
    node_events: list[str] = []
    for chunk in app.stream(current_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"]
            if not interrupt_payload:
                continue
            return StreamOutcome(
                interrupted=True,
                question=interrupt_payload[0].value,
                node_events=node_events,
            )
        node_events.extend(chunk)

    final_values = app.get_state(config).values
    return StreamOutcome(interrupted=False, node_events=node_events, final_values=final_values)


def stage_for_interrupt(question: Any) -> str:
    if isinstance(question, dict) and question.get("kind") == "overview_discussion":
        return STAGE_AWAITING_OVERVIEW_FEEDBACK
    return STAGE_AWAITING_ANSWER
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/webapp/test_driver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/webapp/driver.py tests/webapp/test_driver.py
git commit -m "feat: add stage_for_interrupt routing helper to webapp driver"
```

---

### Task 9: Webapp UI for the discussion loop

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `stage_for_interrupt` (Task 8); `STAGE_AWAITING_OVERVIEW_FEEDBACK` (Task 7); existing `_drive`, `_config`, `get_compiled_app`, `Command`.
- Produces: `_render_awaiting_overview_feedback()`; `main()` dispatches to it when `stage == STAGE_AWAITING_OVERVIEW_FEEDBACK`.

This task has no dedicated automated test: the existing `tests/test_app.py`
only exercises the setup screen and the config-error screen via
`AppTest.from_file` (see its docstring comment on `get_compiled_app` caching)
— it does not click "Start" or drive the graph through any interrupt,
including the pre-existing interview one, because doing so would require
either a real Ollama connection or swapping `get_compiled_app`'s cached
workflow for a `FakeChatModel`-backed one mid-`AppTest`, which this codebase
doesn't currently have infrastructure for. This task matches that existing
scope boundary rather than inventing new Streamlit test infrastructure
in passing. Correctness of the underlying state machine is already covered
by Task 5's graph-level tests and Task 8's `stage_for_interrupt` unit tests;
this task is purely the rendering glue.

- [ ] **Step 1: Update imports**

In `app.py`, update the `state_keys` import block:

```python
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
```

Update the `driver` import:

```python
from src.webapp.driver import run_single_pass, stage_for_interrupt
```

- [ ] **Step 2: Update `_drive` to route by payload shape**

Replace the `if outcome.interrupted:` branch inside `_drive`:

```python
        if outcome.interrupted:
            st.session_state[PENDING_QUESTION] = outcome.question
            st.session_state[STAGE] = stage_for_interrupt(outcome.question)
```

- [ ] **Step 3: Add the render function**

Add a new function right after `_render_awaiting_answer`:

```python
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
```

- [ ] **Step 4: Wire the new stage into `main()`'s dispatch**

Replace the `elif stage == STAGE_AWAITING_ANSWER:` block in `main()`:

```python
    elif stage == STAGE_AWAITING_ANSWER:
        _render_awaiting_answer()
    elif stage == STAGE_AWAITING_OVERVIEW_FEEDBACK:
        _render_awaiting_overview_feedback()
```

- [ ] **Step 5: Run the full existing webapp test suite to confirm no regression**

Run: `pytest tests/test_app.py tests/webapp -v`
Expected: All PASS (these tests don't reach the new code path, but must
still pass since `app.py` is imported/executed by `AppTest.from_file`)

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add webapp UI for overview discussion loop"
```

---

### Task 10: Docs sync and full verification

**Files:**
- Modify: `claude.md` (this repo's `CLAUDE.md`, tracked in git per commit `dde34d8`)

**Interfaces:** None — documentation and verification only.

- [ ] **Step 1: Update the pipeline description in `claude.md`**

Find the line describing the 12-node pipeline (in the `## Pipeline (LangGraph nodes)` section) and update it:

Old:
```
`src/graph/workflow.py` wires 12 nodes (11 pipeline stages; the interview
stage is split across two nodes for the interrupt boundary):

`ingest -> interview_ask -> [interview_wait <-> interview_ask]* -> researcher -> outliner -> writer -> reviewer -> [writer <-> reviewer]* -> overview -> character_bible -> location_bible -> bible_reviewer -> [character_bible <-> bible_reviewer]* -> finalize`
```

New:
```
`src/graph/workflow.py` wires 14 nodes (12 pipeline stages; the interview
and overview-discussion stages are each split across two nodes for their
interrupt boundaries):

`ingest -> interview_ask -> [interview_wait <-> interview_ask]* -> researcher -> outliner -> writer -> reviewer -> [writer <-> reviewer]* -> overview -> [overview_discussion_wait <-> overview_revise]* -> character_bible -> location_bible -> bible_reviewer -> [character_bible <-> bible_reviewer]* -> finalize`
```

Find the bullet describing `overview` in the same section and update it:

Old:
```
* `overview`: single one-shot LLM call (no revision loop) once the draft passes review - writes a short story description plus production tech specs (duration, frame format, aspect ratio, camera, lenses) to `Overview.md`.
```

New:
```
* `overview`: single one-shot LLM call once the draft passes review, producing a pre-result story description plus production tech specs (duration, frame format, aspect ratio, camera, lenses). Non-autonomous runs then pause at `overview_discussion_wait` for free-form human feedback; `overview_revise` regenerates the overview against that feedback and loops back to `overview_discussion_wait` until the user accepts (no revision cap - the human decides when to stop), then the pipeline proceeds to `character_bible`. Autonomous runs skip straight from `overview` to `character_bible`. Whatever `state["overview"]` holds when the loop exits is written to `Overview.md`.
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v --cov=src`
Expected: All tests PASS, including every test added/modified in Tasks 1-9.

- [ ] **Step 3: Run lint and format checks**

Run: `ruff check src tests main.py app.py && ruff format --check src tests main.py app.py`
Expected: No errors. If `ruff format --check` reports files needing
formatting, run `ruff format src tests main.py app.py` and re-check.

- [ ] **Step 4: Commit**

```bash
git add claude.md
git commit -m "docs: document the overview discussion loop in claude.md"
```
