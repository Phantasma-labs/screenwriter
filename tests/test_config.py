# tests/test_config.py
from __future__ import annotations

import pytest

from src.config import ConfigError, get_embeddings, get_llm, load_settings


def test_load_settings_uses_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_BASE_URL", raising=False)
    settings = load_settings()
    assert settings.ollama_model_name == "deepseek-v4.1-flash:cloud"
    assert settings.ollama_base_url == "https://ollama.com"
    assert settings.ollama_api_key == ""
    assert settings.ollama_embed_base_url == "http://localhost:11434"
    assert settings.rag_chunk_size == 1000
    assert settings.rag_chunk_overlap == 150
    assert settings.max_interview_questions == 5
    assert settings.review_pass_score == 8.0


def test_load_settings_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "custom-model:cloud")
    monkeypatch.setenv("RAG_TOP_K", "9")
    settings = load_settings()
    assert settings.ollama_model_name == "custom-model:cloud"
    assert settings.rag_top_k == 9


def test_get_llm_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_llm()


def test_get_llm_builds_chat_ollama_with_configured_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    llm = get_llm()
    assert llm.model == "deepseek-v4.1-flash:cloud"
    assert llm.base_url == "https://ollama.com"


def test_get_embeddings_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    embeddings = get_embeddings()
    assert embeddings.model == "nomic-embed-text"


def test_get_embeddings_defaults_to_local_daemon(monkeypatch):
    monkeypatch.delenv("OLLAMA_EMBED_BASE_URL", raising=False)
    embeddings = get_embeddings()
    assert embeddings.base_url == "http://localhost:11434"


def test_get_embeddings_builds_with_configured_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
    embeddings = get_embeddings()
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://localhost:11434"
