# src/graph/nodes/character_bible.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.bible import (
    CharacterBibleEntry,
    parse_character_entries,
    render_character_bible,
)
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills.base import T2I_PROMPT_GUIDELINES

_SYSTEM_PROMPT = (
    "You identify the principal cast of a script - characters significant enough "
    "to warrant a full production bible entry, not every speaking role - and write "
    "one entry per principal character.\n\n"
    "Only include characters who are part of the dramatized story and physically "
    "appear on screen as that story's characters. Never write an entry for a "
    "voice-only role (e.g. an off-screen NARRATOR or a V.O.-only voice), a "
    "sound/music cue (e.g. MUSIC, SFX, OST), or a real-world documentary "
    "apparatus role - a host, narrator, or expert/historian interview subject "
    "who provides narration or framing commentary - even if that person "
    "physically appears on camera for an interview segment. Those roles are "
    "not dramatized characters and do not need a casting or wardrobe "
    "reference.\n\n"
    "The wardrobe_prompt must render the character's full wardrobe and "
    "accessories against a plain, seamless white studio background (no set or "
    "location) - full-body, evenly lit - since it will be used later as a clean "
    "visual reference.\n\n" + T2I_PROMPT_GUIDELINES
)

_RESPONSE_INSTRUCTION = (
    'Respond with a JSON array only: [{"name": str, "role": str, "appearance": str, '
    '"personality": str, "voice": str, "backstory": str, "headshot_prompt": str, '
    '"contact_sheet_prompt": str, "wardrobe_prompt": str}, ...]'
)

_EXCLUDED_ROLE_PATTERNS = (
    "narrator",
    "historian",
    "voice-only",
    "voice only",
    "off-screen",
    "off screen",
    "no physical form",
)


def _is_documentary_apparatus(entry: CharacterBibleEntry) -> bool:
    haystack = f"{entry.role} {entry.appearance}".lower()
    return any(pattern in haystack for pattern in _EXCLUDED_ROLE_PATTERNS)


def make_character_bible_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def character_bible_node(state: ScreenplayState) -> NodeUpdate:
        is_revision = bool(state["bible_review_feedback"])
        feedback_note = (
            f"\n\nAddress this reviewer feedback:\n{state['bible_review_feedback']}"
            if is_revision
            else ""
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"{_RESPONSE_INSTRUCTION}{feedback_note}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entries = parse_character_entries(str(response.content))
        kept_entries = [entry for entry in entries if not _is_documentary_apparatus(entry)]
        update: NodeUpdate = {
            "character_bible": render_character_bible(kept_entries),
            "character_names": [entry.name for entry in entries],
        }
        if is_revision:
            update["bible_revision_count"] = state["bible_revision_count"] + 1
        return update

    return character_bible_node
