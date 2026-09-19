# tests/test_search_fallback.py
from __future__ import annotations

from src.config import Settings
from src.tools.search import SearchResult, search


def _settings(tavily_api_key: str | None = None) -> Settings:
    return Settings(
        ollama_model_name="m",
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        ollama_embed_model="e",
        tavily_api_key=tavily_api_key,
        rag_chunk_size=1000,
        rag_chunk_overlap=150,
        rag_top_k=5,
        rag_min_chars_to_index=4000,
        max_interview_questions=5,
        review_pass_score=8.0,
        max_bible_revisions=1,
    )


def test_search_uses_tavily_when_key_present():
    calls = []

    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        calls.append((query, api_key))
        return [SearchResult(title="T", url="u", snippet="s")]

    def fake_ddgs(query: str) -> list[SearchResult]:
        raise AssertionError("DDGS should not be called when Tavily succeeds")

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="T", url="u", snippet="s")]
    assert calls == [("test query", "tavily-key")]


def test_search_falls_back_to_ddgs_when_no_tavily_key():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise AssertionError("Tavily should not be called without a key")

    def fake_ddgs(query: str) -> list[SearchResult]:
        return [SearchResult(title="D", url="u2", snippet="s2")]

    results = search(
        "test query", settings=_settings(None), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="D", url="u2", snippet="s2")]


def test_search_falls_back_to_ddgs_when_tavily_raises():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise RuntimeError("network down")

    def fake_ddgs(query: str) -> list[SearchResult]:
        return [SearchResult(title="D", url="u2", snippet="s2")]

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="D", url="u2", snippet="s2")]


def test_search_returns_empty_when_both_fail():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise RuntimeError("network down")

    def fake_ddgs(query: str) -> list[SearchResult]:
        raise RuntimeError("ddgs also down")

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == []
