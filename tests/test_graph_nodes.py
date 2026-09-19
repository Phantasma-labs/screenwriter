# tests/test_graph_nodes.py
from __future__ import annotations

from src.config import Settings
from src.graph.nodes.character_bible import make_character_bible_node
from src.graph.nodes.ingest import make_ingest_node
from src.graph.nodes.outliner import make_outliner_node
from src.graph.nodes.researcher import make_researcher_node
from src.graph.nodes.reviewer import make_reviewer_node
from src.graph.nodes.writer import make_writer_node
from src.graph.state import new_initial_state
from src.rag.store import clear_store, get_store
from src.tools.search import SearchResult
from tests.conftest import FakeChatModel


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def _settings(**overrides: object) -> Settings:
    base = dict(
        ollama_model_name="m",
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        ollama_embed_model="e",
        tavily_api_key=None,
        rag_chunk_size=50,
        rag_chunk_overlap=5,
        rag_top_k=5,
        rag_min_chars_to_index=20,
        max_interview_questions=5,
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


def test_ingest_node_no_files_skips_parsing():
    node = make_ingest_node(settings=_settings())
    result = node(_state(file_paths=[]))
    assert result == {"parsed_context": "", "rag_indexed": False, "status": "ingested"}


def test_ingest_node_short_context_skips_rag(tmp_path):
    md_file = tmp_path / "short.md"
    md_file.write_text("Short note.", encoding="utf-8")
    node = make_ingest_node(settings=_settings(rag_min_chars_to_index=1000))
    result = node(_state(file_paths=[str(md_file)]))
    assert result["rag_indexed"] is False
    assert "Short note." in result["parsed_context"]


def test_ingest_node_long_context_indexes_into_rag(tmp_path):
    md_file = tmp_path / "long.md"
    md_file.write_text("word " * 50, encoding="utf-8")
    node = make_ingest_node(
        embeddings=_FakeEmbeddings(), settings=_settings(rag_min_chars_to_index=20)
    )
    result = node(_state(file_paths=[str(md_file)]))
    try:
        assert result["rag_indexed"] is True
        assert result["rag_run_id"]
        assert get_store(result["rag_run_id"]) is not None
    finally:
        clear_store(result.get("rag_run_id", ""))


def test_researcher_node_disabled_returns_empty_notes():
    node = make_researcher_node(
        enabled=False, search_fn=lambda q: [SearchResult(title="x", url="u", snippet="s")]
    )
    result = node(_state(topic="anything"))
    assert result["research_notes"] == ""


def test_researcher_node_enabled_formats_results():
    def fake_search(query: str) -> list[SearchResult]:
        assert query == "space tourism"
        return [SearchResult(title="Title", url="http://x", snippet="Snippet text")]

    node = make_researcher_node(enabled=True, search_fn=fake_search)
    result = node(_state(topic="space tourism"))
    assert "Title" in result["research_notes"]
    assert "Snippet text" in result["research_notes"]
    assert "http://x" in result["research_notes"]


def test_outliner_node_returns_llm_output_as_outline():
    llm = FakeChatModel(responses=["1. Hook\n2. Climax\n3. Resolution"])
    node = make_outliner_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(_state(topic="A retired detective solves crimes via voicemail"))
    assert result["outline"] == "1. Hook\n2. Climax\n3. Resolution"
    assert result["status"] == "outlined"


def test_outliner_node_passes_retrieved_chunks_into_prompt():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=["outline text"])
    node = make_outliner_node(
        llm=llm,
        retrieve_fn=lambda run_id, query, k: ["Retrieved chunk about the detective's past."],
    )
    node(_state(topic="t", rag_run_id="run-1"))
    human_content = str(captured_messages[0][-1].content)
    assert "Retrieved chunk about the detective's past." in human_content


def test_writer_node_first_draft_does_not_bump_revision_count():
    llm = FakeChatModel(responses=["INT. ROOM - DAY\n\nShe waits.\n"])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(_state(outline="1. Hook", review_feedback=""))
    assert result["draft"] == "INT. ROOM - DAY\n\nShe waits.\n"
    assert "revision_count" not in result


def test_writer_node_revision_pass_bumps_revision_count():
    llm = FakeChatModel(responses=["INT. ROOM - DAY\n\nShe answers.\n"])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(
        _state(outline="1. Hook", review_feedback="Trim the action lines.", revision_count=0)
    )
    assert result["revision_count"] == 1


def test_writer_node_uses_dual_column_instruction_for_commercial_skill():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=['[{"timecode": "0:00", "visual": "v", "audio": "a"}]'])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    node(_state(skill="commercial", outline="1. Hook"))
    human_content = str(captured_messages[0][-1].content)
    assert "JSON array" in human_content


def test_reviewer_node_parses_valid_json():
    llm = FakeChatModel(
        responses=[
            '{"score": 9.2, "passed": true, "critique": "Strong draft.", '
            '"actionable_revisions": []}'
        ]
    )
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="INT. ROOM - DAY\n\nShe waits."))
    assert result["review_score"] == 9.2
    assert "Strong draft." in result["review_feedback"]


def test_reviewer_node_includes_actionable_revisions_in_feedback():
    llm = FakeChatModel(
        responses=[
            '{"score": 5.0, "passed": false, "critique": "Needs work.", '
            '"actionable_revisions": ["Trim action lines", "Fix slugline case"]}'
        ]
    )
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["review_score"] == 5.0
    assert "Trim action lines" in result["review_feedback"]
    assert "Fix slugline case" in result["review_feedback"]


def test_reviewer_node_falls_back_gracefully_on_unparseable_response():
    llm = FakeChatModel(responses=["I refuse to output JSON today."])
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["review_score"] == 0.0
    assert "could not be parsed" in result["review_feedback"]


_CHARACTER_JSON = (
    '[{"name": "Jane", "role": "Protagonist", "appearance": "Sharp business attire.", '
    '"personality": "Relentless.", "voice": "Clipped.", "backstory": "Ex-detective.", '
    '"headshot_prompt": "Create a headshot of Jane.", '
    '"contact_sheet_prompt": "Create a contact sheet of Jane.", '
    '"wardrobe_prompt": "Create a wardrobe shot of Jane."}]'
)


def test_character_bible_node_renders_entries_from_valid_json():
    llm = FakeChatModel(responses=[_CHARACTER_JSON])
    node = make_character_bible_node(llm=llm)
    result = node(_state(draft="INT. OFFICE - DAY\n\nJANE stares at the phone."))
    assert "Jane" in result["character_bible"]
    assert "Headshot Prompt" in result["character_bible"]


def test_character_bible_node_empty_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_character_bible_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["character_bible"] == "# Character Bible\n\nNo principal characters identified.\n"


def test_character_bible_node_includes_prior_feedback_when_revising():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=[_CHARACTER_JSON])
    node = make_character_bible_node(llm=llm)
    node(_state(draft="draft text", bible_review_feedback="Add more wardrobe detail."))
    human_content = str(captured_messages[0][-1].content)
    assert "Add more wardrobe detail." in human_content
