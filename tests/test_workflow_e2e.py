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
        '{"story_description": "A retired detective solves crimes via voicemail.", '
        '"duration_estimate": "8-10 minutes", "frame_format": "Digital Cinema, 4K", '
        '"aspect_ratio": "2.39:1", "camera": "ARRI Alexa Mini", "lenses": "35mm prime"}',
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
    assert "# Overview" in result["overview"]
    assert "A retired detective solves crimes via voicemail." in result["overview"]
    assert "2.39:1" in result["overview"]


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
        # autonomous=False, so interview_ask always makes one LLM decision call
        # before routing on; this response clears the interview immediately so
        # the run proceeds straight to the outline/draft/review/overview calls
        # the brief's response list otherwise assumes come first.
        '{"has_enough_info": true, "question": ""}',
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


def test_workflow_blank_overview_feedback_reprompts_without_revising():
    responses = [
        '{"has_enough_info": true, "question": ""}',
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
    config = {"configurable": {"thread_id": "test-thread-blank-overview-feedback"}}

    paused = app.invoke(initial_state, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "overview_discussion"
    overview_before_feedback = paused["__interrupt__"][0].value["overview"]
    assert "A phone rings once a year." in overview_before_feedback

    # Submit real feedback: this should revise for real, populating the
    # transcript with one turn.
    revised = app.invoke(
        Command(resume={"feedback": "Make it darker and quieter.", "finish": False}), config
    )
    assert "__interrupt__" in revised
    revised_overview = revised["__interrupt__"][0].value["overview"]
    assert "now darker and quieter" in revised_overview
    calls_after_real_feedback = llm._call_count

    # Blank Enter: no feedback, not finished, with a PRIOR feedback turn
    # already sitting in the transcript. This is the actual bug scenario -
    # the old code routed to overview_revise unconditionally, which reread
    # transcript[-1]["feedback"] (the stale prior feedback) and burned an
    # LLM call silently re-revising against input the user didn't resubmit.
    # The fix must re-pause at overview_discussion_wait immediately (a
    # self-loop re-prompt) WITHOUT ever calling overview_revise's LLM again,
    # and the overview must be unchanged from the real-feedback revision.
    paused_again = app.invoke(Command(resume={"feedback": "", "finish": False}), config)
    assert "__interrupt__" in paused_again
    assert paused_again["__interrupt__"][0].value["kind"] == "overview_discussion"
    assert paused_again["__interrupt__"][0].value["overview"] == revised_overview
    assert llm._call_count == calls_after_real_feedback
    assert paused_again["overview_discussion_transcript"] == [
        {
            "feedback": "Make it darker and quieter.",
            "overview_snapshot": overview_before_feedback,
        }
    ]

    # Blank Enter again, but this time finished: proceeds to finalize.
    resumed = app.invoke(Command(resume={"feedback": "", "finish": True}), config)
    assert "__interrupt__" not in resumed
    assert resumed["status"] == "finalized"
    assert resumed["overview_discussion_transcript"] == [
        {
            "feedback": "Make it darker and quieter.",
            "overview_snapshot": overview_before_feedback,
        }
    ]


def test_workflow_autonomous_from_interview_also_bypasses_overview_discussion():
    responses = [
        '{"has_enough_info": false, "question": "What tone should this have?"}',
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
    config = {"configurable": {"thread_id": "test-thread-autonomous-from-interview"}}

    paused = app.invoke(initial_state, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == "What tone should this have?"

    result = app.invoke(Command(resume={"answer": "some answer", "autonomous": True}), config)

    assert "__interrupt__" not in result
    assert result["status"] == "finalized"
    assert result["overview_discussion_transcript"] == []


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
        '{"story_description": "A retired detective solves crimes via voicemail.", '
        '"duration_estimate": "8-10 minutes", "frame_format": "Digital Cinema, 4K", '
        '"aspect_ratio": "2.39:1", "camera": "ARRI Alexa Mini", "lenses": "35mm prime"}',
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
