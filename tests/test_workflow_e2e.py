# tests/test_workflow_e2e.py
from __future__ import annotations

from langgraph.types import Command

from src.graph.state import new_initial_state
from src.graph.workflow import build_workflow
from src.tools.search import SearchResult
from tests.conftest import FakeChatModel


def test_full_workflow_autonomous_short_film_produces_all_artifacts():
    responses = [
        "1. Hook\n2. Inciting incident\n3. Climax\n4. Resolution",
        "INT. OFFICE - DAY\n\nJANE stares at the phone.\n\nJANE\nPick up.\n",
        '{"score": 9.0, "passed": true, "critique": "Solid draft.", "actionable_revisions": []}',
        '[{"name": "Jane", "role": "Protagonist", '
        '"appearance": "Sharp business attire, tired eyes.", '
        '"personality": "Relentless.", "voice": "Clipped, impatient.", '
        '"backstory": "Ex-detective.", '
        '"headshot_prompt": "Create a headshot of Jane, a woman with tired eyes '
        'in sharp business attire.", '
        '"contact_sheet_prompt": "Create a 3x2 contact sheet of Jane in six expressions.", '
        '"wardrobe_prompt": "Create a full-body wardrobe shot of Jane in sharp '
        'business attire."}]',
        '[{"name": "Office", "description": "A cramped, fluorescent-lit detective office.", '
        '"mood": "Tense and claustrophobic.", '
        '"t2i_prompt": "Create a wide shot of a cramped, fluorescent-lit detective '
        'office at night."}]',
        '{"score": 9.0, "passed": true, "critique": "Consistent with the draft.", '
        '"actionable_revisions": []}',
        "Create a moody cinematic poster of a detective silhouetted against office blinds.",
    ]
    llm = FakeChatModel(responses=responses)

    def fake_search(query: str) -> list[SearchResult]:
        return [
            SearchResult(title="Research hit", url="http://example.com", snippet="Useful context")
        ]

    app = build_workflow(enable_search=True, llm=llm, search_fn=fake_search)

    initial_state = new_initial_state(
        topic="A retired detective solves crimes via voicemail",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=True,
    )
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = app.invoke(initial_state, config)

    assert "__interrupt__" not in result
    assert result["status"] == "finalized"
    assert "INT. OFFICE - DAY" in result["fountain_script"]
    assert "Key Art T2I Prompt" in result["screenplay_markdown"]
    assert "# Character Bible" in result["character_bible"]
    assert "Jane" in result["character_bible"]
    assert "Headshot Prompt" in result["character_bible"]
    assert "# Location Bible" in result["location_bible"]
    assert "Office" in result["location_bible"]


def test_workflow_interview_pauses_for_human_input_then_resumes():
    responses = [
        '{"has_enough_info": false, "question": "What tone should this have?"}',
        '{"has_enough_info": true, "question": ""}',
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
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

    resumed = app.invoke(
        Command(resume={"answer": "Melancholy and quiet.", "autonomous": False}), config
    )

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "finalized"
    assert resumed["interview_transcript"] == [
        {"question": "What tone should this have?", "answer": "Melancholy and quiet."}
    ]


def test_workflow_bible_revise_loop_runs_twice_then_finalizes():
    character_json_pass_1 = (
        '[{"name": "Jane", "role": "Protagonist", "appearance": "Sharp business attire.", '
        '"personality": "Relentless.", "voice": "Clipped.", "backstory": "Ex-detective.", '
        '"headshot_prompt": "Create a headshot of Jane.", '
        '"contact_sheet_prompt": "Create a contact sheet of Jane.", '
        '"wardrobe_prompt": "Create a wardrobe shot of Jane."}]'
    )
    character_json_pass_2 = (
        '[{"name": "Jane", "role": "Protagonist", "appearance": "Sharp business attire.", '
        '"personality": "Relentless.", "voice": "Clipped.", '
        '"backstory": "Ex-detective, revised with more concrete detail.", '
        '"headshot_prompt": "Create a headshot of Jane.", '
        '"contact_sheet_prompt": "Create a contact sheet of Jane.", '
        '"wardrobe_prompt": "Create a wardrobe shot of Jane."}]'
    )
    location_json_pass_1 = (
        '[{"name": "Office", "description": "A cramped detective office.", '
        '"mood": "Tense.", "t2i_prompt": "Create a wide shot of a cramped office."}]'
    )
    location_json_pass_2 = (
        '[{"name": "Office", "description": "A cramped detective office, revised.", '
        '"mood": "Tense.", "t2i_prompt": "Create a wide shot of a cramped office."}]'
    )
    responses = [
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
        character_json_pass_1,
        location_json_pass_1,
        '{"score": 3.0, "passed": false, "critique": "Needs more concrete detail.", '
        '"actionable_revisions": ["Add more backstory detail."]}',
        character_json_pass_2,
        location_json_pass_2,
        '{"score": 9.0, "passed": true, "critique": "Consistent now.", "actionable_revisions": []}',
        "Create a moody cinematic poster of a detective silhouetted against office blinds.",
    ]
    llm = FakeChatModel(responses=responses)
    app = build_workflow(enable_search=False, llm=llm, search_fn=lambda q: [])

    initial_state = new_initial_state(
        topic="A retired detective solves crimes via voicemail",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=2,
        autonomous=True,
    )
    config = {"configurable": {"thread_id": "test-thread-bible-revise"}}
    result = app.invoke(initial_state, config)

    assert "__interrupt__" not in result
    assert result["status"] == "finalized"
    assert result["bible_revision_count"] == 1
    assert "revised with more concrete detail" in result["character_bible"]
    assert "revised" in result["location_bible"]
