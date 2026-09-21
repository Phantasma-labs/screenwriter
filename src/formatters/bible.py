# src/formatters/bible.py
from __future__ import annotations

from pydantic import BaseModel

from src.formatters.t2i import t2i_box
from src.utils import extract_json_array


class CharacterBibleEntry(BaseModel):
    name: str
    role: str
    appearance: str
    personality: str
    voice: str
    backstory: str
    headshot_prompt: str
    contact_sheet_prompt: str
    wardrobe_prompt: str


class LocationBibleEntry(BaseModel):
    name: str
    description: str
    mood: str
    t2i_prompt: str


def parse_character_entries(raw: str) -> list[CharacterBibleEntry]:
    return extract_json_array(raw, CharacterBibleEntry)


def parse_location_entries(raw: str) -> list[LocationBibleEntry]:
    return extract_json_array(raw, LocationBibleEntry)


def render_character_bible(entries: list[CharacterBibleEntry]) -> str:
    if not entries:
        return "# Character Bible\n\nNo principal characters identified.\n"
    sections = ["# Character Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name} — {entry.role}\n")
        sections.append(f"**Appearance:** {entry.appearance}\n")
        sections.append(f"**Personality:** {entry.personality}\n")
        sections.append(f"**Voice:** {entry.voice}\n")
        sections.append(f"**Backstory:** {entry.backstory}\n")
        sections.append(t2i_box("Headshot Prompt", entry.headshot_prompt))
        sections.append(t2i_box("Contact Sheet Prompt", entry.contact_sheet_prompt))
        sections.append(t2i_box("Wardrobe & Accessories Prompt", entry.wardrobe_prompt))
    return "\n".join(sections)


def render_location_bible(entries: list[LocationBibleEntry]) -> str:
    if not entries:
        return "# Location Bible\n\nNo locations identified.\n"
    sections = ["# Location Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name}\n")
        sections.append(f"**Description:** {entry.description}\n")
        sections.append(f"**Mood:** {entry.mood}\n")
        sections.append(t2i_box("Location T2I Prompt", entry.t2i_prompt))
    return "\n".join(sections)
