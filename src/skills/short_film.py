# src/skills/short_film.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

SHORT_FILM = SkillProfile(
    name="short_film",
    display_name="Short Film",
    output_format=OutputFormat.FOUNTAIN,
    persona=(
        "You are an award-winning short film screenwriter. Write in Master Scene "
        "Format: show, don't tell; every line of dialogue carries subtext; action "
        "lines stay lean and visual."
    ),
    outline_template=[
        "Opening image / hook",
        "Inciting incident",
        "Escalating complication",
        "Turning point",
        "Climax",
        "Resolution / final image",
    ],
    review_criteria=[
        "Scene headings are ALL CAPS and correctly formatted (INT./EXT. LOCATION - TIME)",
        "Action blocks are 4 lines or fewer per paragraph",
        "Dialogue sounds spoken, not expository",
        "Parentheticals are used sparingly and only when meaning isn't clear from context",
        "Subtext: characters rarely state their goal outright",
        "Visual storytelling: the camera could film every action line as written",
    ],
)
