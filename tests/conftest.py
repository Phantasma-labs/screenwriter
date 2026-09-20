# tests/conftest.py
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


@pytest.fixture(autouse=True)
def _default_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "deepseek-v4.1-flash:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")


class FakeChatModel(BaseChatModel):
    """Deterministic offline chat model double. Returns `responses` in order;
    repeats the last one once exhausted. `_call_count` lets tests assert how
    many times `.invoke()` was called."""

    responses: list[str]
    _call_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        index = min(self._call_count, len(self.responses) - 1)
        content = self.responses[index]
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
