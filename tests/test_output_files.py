# tests/test_output_files.py
from __future__ import annotations

from src.output_files import build_output_files


def _result() -> dict[str, str]:
    return {
        "overview": "OVERVIEW TEXT",
        "fountain_script": "FOUNTAIN TEXT",
        "screenplay_markdown": "MARKDOWN TEXT",
        "character_bible": "CHAR BIBLE",
        "location_bible": "LOC BIBLE",
    }


def test_build_output_files_maps_to_five_fixed_names():
    outputs = build_output_files(_result())
    assert outputs == {
        "Overview.md": "OVERVIEW TEXT",
        "Script.md": "FOUNTAIN TEXT",
        "Screenplay.md": "MARKDOWN TEXT",
        "CharacterBible.md": "CHAR BIBLE",
        "LocationBible.md": "LOC BIBLE",
    }
