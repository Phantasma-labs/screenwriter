# Overview Discussion Loop — Design

## Summary

Insert a human-in-the-loop discussion stage between the existing `overview`
node and `character_bible` node. Today `overview` produces a final overview
in one shot and the graph proceeds straight to the bibles. After this change,
`overview` produces a **pre-result** that the user can iteratively critique
in free-form natural language; the LLM regenerates the overview against that
feedback each round until the user explicitly accepts it (`/finish` or
equivalent), at which point the graph proceeds to `character_bible` ->
`location_bible` -> `bible_reviewer` -> `finalize` exactly as it does today.
`--autonomous` runs skip the discussion entirely, consistent with how they
already skip the interview stage.

This reuses the existing `interrupt()`/`Command(resume=...)` mechanism that
`interview_ask`/`interview_wait` already established, and reuses the
`parse_overview`/`render_overview` machinery the current `overview` node
already uses.

## Pipeline changes

Two new nodes are inserted between `overview` and `character_bible`, mirroring
the `interview_ask <-> interview_wait` split (one node is a pure interrupt
boundary, the other does the LLM work) but with the roles reversed: here the
human critiques and the LLM revises.

```
... reviewer -> overview -> [overview_discussion_wait <-> overview_revise]* -> character_bible -> ...
```

* **`overview_discussion_wait`** (interrupt boundary, no LLM call): calls
  `interrupt({"kind": "overview_discussion", "overview": state["overview"]})`
  and blocks for CLI/webapp input. Resume payload is
  `{"feedback": str, "finish": bool}`. Appends an `OverviewDiscussionTurn` to
  the transcript (skipped when `feedback` is empty, i.e. a bare accept with no
  critique) and records whether the user is done.
* **`overview_revise`** (LLM call, no interrupt): takes the current overview,
  the approved draft, and the latest feedback, and regenerates the overview
  using the same `parse_overview`/`render_overview`/`FALLBACK_OVERVIEW`
  fallback path the existing `overview` node uses. Updates `state["overview"]`
  in place (same field, not a separate "final" field).

New edges (`src/graph/workflow.py`):

* `overview` -> conditional on `route_after_overview`:
  * `autonomous` -> `character_bible` (bypasses discussion entirely)
  * else -> `overview_discussion_wait`
* `overview_discussion_wait` -> conditional on `route_after_overview_discussion_wait`:
  * `overview_discussion_finished` -> `character_bible`
  * else -> `overview_revise`
* `overview_revise` -> `overview_discussion_wait` (always loops back)

No revision cap — the loop continues until the user explicitly finishes.
This intentionally differs from the writer/reviewer and bible loops, which
are capped by `max_revisions`/`max_bible_revisions`; there is no equivalent
`max_overview_discussion_turns` setting, since the human is the one deciding
when to stop, not an automated score threshold.

## State changes (`src/graph/state.py`)

```python
class OverviewDiscussionTurn(TypedDict):
    feedback: str
    overview_snapshot: str  # the overview version this feedback was given on

# ScreenplayState additions:
overview_discussion_transcript: list[OverviewDiscussionTurn]
overview_discussion_finished: bool
```

`new_initial_state` initializes `overview_discussion_transcript=[]` and
`overview_discussion_finished=False`.

Per the state-management convention already in this codebase, each node only
writes the fields it owns: `overview_discussion_wait` owns
`overview_discussion_transcript` and `overview_discussion_finished`;
`overview_revise` owns `overview` (same field `overview` node already owns —
they never run concurrently, so no ownership conflict).

## Edges (`src/graph/edges.py`)

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

Both follow the existing style in this file: plain functions taking
`ScreenplayState`, no `Settings` dependency needed since there's no numeric
threshold involved.

## Node implementation notes

`overview_discussion_wait` (new file `src/graph/nodes/overview_discussion.py`,
alongside `overview_revise` — one module, two functions/factories, matching
how `interviewer.py` houses both `interview_ask` and `interview_wait`):

```python
def overview_discussion_wait_node(state: ScreenplayState) -> NodeUpdate:
    resumed = interrupt({"kind": "overview_discussion", "overview": state["overview"]})
    feedback = resumed.get("feedback", "")
    finish = bool(resumed.get("finish", False))
    transcript = state["overview_discussion_transcript"]
    if feedback:
        turn: OverviewDiscussionTurn = {"feedback": feedback, "overview_snapshot": state["overview"]}
        transcript = transcript + [turn]
    return {
        "overview_discussion_transcript": transcript,
        "overview_discussion_finished": finish,
    }
```

`overview_revise` follows the existing `overview_node` shape (same system
prompt intent, same `_RESPONSE_INSTRUCTION` JSON contract), with the human
message additionally carrying the current overview text and the latest
feedback entry so the LLM revises rather than starting from scratch. On parse
failure it keeps the existing `state["overview"]` unchanged rather than
falling back to `FALLBACK_OVERVIEW` (unlike the first-pass `overview` node,
here a fallback would silently discard a perfectly good prior version).

## CLI changes (`main.py`)

`run_interactive`'s interrupt handling currently assumes the interrupt value
is always a bare question string — that's the interview node's contract and
is left untouched. It's extended to branch on payload shape:

```python
value = interrupt_payload[0].value
if isinstance(value, dict) and value.get("kind") == "overview_discussion":
    print("[OVERVIEW PRE-RESULT]")
    print(value["overview"])
    feedback = input("Feedback (or /finish to accept and continue): ")
    finish = is_finish_command(feedback)
    current_input = Command(resume={"feedback": "" if finish else feedback, "finish": finish})
else:
    question = value
    print(f"[INTERVIEWING] {question}")
    answer = input("> ")
    current_input = Command(resume={"answer": answer, "autonomous": is_finish_command(answer)})
```

`STATUS_BADGES` gains entries for `overview_discussion_wait` and
`overview_revise` (e.g. `[OVERVIEW DISCUSSION]` / `[REVISING OVERVIEW]`).

## Webapp changes (`app.py`, `src/webapp/`)

* `StreamOutcome.question` (`src/webapp/driver.py`) widens from `str | None`
  to `Any | None` — it was already an opaque pass-through of the interrupt
  value past the boundary, this just makes the type honest now that it can be
  a dict.
* `src/webapp/state_keys.py` gains `STAGE_AWAITING_OVERVIEW_FEEDBACK`.
* `_drive` in `app.py` inspects the interrupt payload's `kind` and routes to
  the new stage instead of `STAGE_AWAITING_ANSWER` when it's
  `overview_discussion`.
* New `_render_awaiting_overview_feedback()` renders the pre-result overview
  (`st.markdown`) plus a feedback text area and two buttons: "Revise"
  (`Command(resume={"feedback": text, "finish": False})`) and "Accept and
  continue" (`Command(resume={"feedback": "", "finish": True})`) — mirroring
  the existing "Submit answer" / "Finish now" pair in
  `_render_awaiting_answer`.
* `main()`'s stage dispatch gains the new stage branch.

## Testing plan (TDD)

Tests written first, per this repo's workflow, all offline per existing
conventions (`FakeChatModel`, no live calls):

* `tests/test_graph_nodes.py` — `overview_discussion_wait_node` (interrupt
  contract, transcript append/skip-on-empty-feedback, `finish` propagation)
  and `overview_revise` factory (regenerates `overview`, keeps prior value on
  parse failure).
* `tests/test_graph_edges.py` — `route_after_overview` (autonomous vs not)
  and `route_after_overview_discussion_wait` (finished vs not).
* `tests/test_workflow.py` / `tests/test_workflow_e2e.py` — full graph
  traversal through one or more discussion rounds using `FakeChatModel` and
  simulated `Command(resume=...)`, confirming autonomous runs bypass the loop
  entirely and non-autonomous runs stop exactly at
  `overview_discussion_wait`.
* `tests/test_main.py` — CLI branch dispatch on interrupt payload shape.
* `tests/test_app.py` / `tests/webapp/test_driver.py` — webapp stage
  routing and the new render path.

## Out of scope

* No turn cap / `max_overview_discussion_turns` setting (explicit decision —
  the human decides when to stop).
* No changes to `finalize`, the bible loop, or output file names/shape —
  `Overview.md` still ends up holding whatever `state["overview"]` is when
  the discussion loop exits.
* No changes to the interview stage's interrupt payload contract (kept as a
  bare string) to avoid touching already-passing tests unrelated to this
  feature.
