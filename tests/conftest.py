# tests/conftest.py
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "deepseek-v4.1-flash:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
