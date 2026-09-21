# tests/test_workflow.py
from __future__ import annotations

import src.graph.workflow as workflow_module
from src.config import Settings
from src.graph.workflow import build_workflow


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


def _settings(**overrides: object) -> Settings:
    base = dict(
        ollama_model_name="m",
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        ollama_embed_model="e",
        ollama_embed_base_url="http://localhost:11434",
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


def test_build_workflow_threads_settings_rag_top_k_into_outliner_and_writer(monkeypatch):
    captured_outliner_kwargs: dict[str, object] = {}
    captured_writer_kwargs: dict[str, object] = {}

    def fake_make_outliner_node(**kwargs: object):
        captured_outliner_kwargs.update(kwargs)
        return lambda state: {}

    def fake_make_writer_node(**kwargs: object):
        captured_writer_kwargs.update(kwargs)
        return lambda state: {}

    monkeypatch.setattr(workflow_module, "make_outliner_node", fake_make_outliner_node)
    monkeypatch.setattr(workflow_module, "make_writer_node", fake_make_writer_node)

    build_workflow(enable_search=False, settings=_settings(rag_top_k=3))

    assert captured_outliner_kwargs["rag_top_k"] == 3
    assert captured_writer_kwargs["rag_top_k"] == 3
