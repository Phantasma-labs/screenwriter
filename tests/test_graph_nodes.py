# tests/test_graph_nodes.py
from __future__ import annotations

from src.config import Settings
from src.graph.nodes.ingest import make_ingest_node
from src.graph.nodes.researcher import make_researcher_node
from src.graph.state import new_initial_state
from src.rag.store import clear_store, get_store
from src.tools.search import SearchResult


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
