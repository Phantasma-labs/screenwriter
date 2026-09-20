# tests/test_app.py
from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _clear_resource_cache():
    # get_compiled_app is @st.cache_resource, which caches at process scope
    # (not per AppTest run) - each test needs a clean slate so a config error
    # in one test isn't masked by a workflow another test already cached.
    st.cache_resource.clear()


def test_app_renders_setup_widgets_when_configured(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    at = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("Topic" in text_input.label for text_input in at.text_area)
    assert any(button.label == "Start" for button in at.button)


def test_app_shows_config_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    at = AppTest.from_file(_APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("Configuration error" in err.value for err in at.error)
    assert len(at.button) == 0
