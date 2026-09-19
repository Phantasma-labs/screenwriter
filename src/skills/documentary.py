# src/skills/documentary.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

DOCUMENTARY = SkillProfile(
    name="documentary",
    display_name="Documentary",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a documentary film writer-producer. You script in Dual-Column "
        "Audio/Visual format: the VISUAL column carries B-roll, archival footage "
        "cues, and interview framing; the AUDIO column carries voice-over (VO), "
        "interview subject dialogue, and ambient sound notes."
    ),
    outline_template=[
        "Cold open / hook",
        "Thesis / question the film explores",
        "Key interview subjects introduced",
        "Archival + B-roll evidence beats",
        "Turning point / complication in the narrative",
        "Resolution and closing VO",
    ],
    review_criteria=[
        "Every row has both a VISUAL and an AUDIO cue - no empty columns",
        "Archival footage is explicitly labeled as archival, not implied",
        "VO reads at a natural spoken pace, not written prose",
        "Interview questions (if included) are open-ended, not leading",
        "B-roll cues are specific and shootable, not abstract",
    ],
)
