# tests/test_formatters_overview.py
from __future__ import annotations

from src.formatters.overview import FALLBACK_OVERVIEW, OverviewEntry, render_overview


def _entry() -> OverviewEntry:
    return OverviewEntry(
        story_description="A retired detective solves crimes left on old voicemails.",
        duration_estimate="8-10 minutes",
        frame_format="Digital Cinema, 4K",
        aspect_ratio="2.39:1",
        camera="ARRI Alexa Mini",
        lenses="Zeiss Supreme Primes, 35mm and 50mm",
    )


def test_render_overview_includes_story_description_and_tech_specs():
    rendered = render_overview(_entry())
    assert "# Overview" in rendered
    assert "A retired detective solves crimes left on old voicemails." in rendered
    assert "8-10 minutes" in rendered
    assert "2.39:1" in rendered
    assert "ARRI Alexa Mini" in rendered
    assert "Zeiss Supreme Primes, 35mm and 50mm" in rendered


def test_fallback_overview_has_placeholder_content():
    rendered = render_overview(FALLBACK_OVERVIEW)
    assert "# Overview" in rendered
    assert FALLBACK_OVERVIEW.story_description in rendered
