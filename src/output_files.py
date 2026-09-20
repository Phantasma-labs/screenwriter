# src/output_files.py
from __future__ import annotations

from src.graph.state import ScreenplayState


def build_output_files(result: ScreenplayState) -> dict[str, str]:
    return {
        "Overview.md": result["overview"],
        "Script.md": result["fountain_script"],
        "Screenplay.md": result["screenplay_markdown"],
        "CharacterBible.md": result["character_bible"],
        "LocationBible.md": result["location_bible"],
    }
