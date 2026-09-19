# src/skills/learning_dev.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

LEARNING_DEV = SkillProfile(
    name="learning_dev",
    display_name="Learning & Development",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an instructional designer scripting a corporate training "
        "video. You script in Dual-Column format: the VISUAL column carries "
        "on-screen text (OST) and presenter direction, the AUDIO column carries "
        "narration written to a stated learning objective, with interactive "
        "pause prompts inserted where a learner should reflect or act."
    ),
    outline_template=[
        "Learning objective stated up front",
        "Concept introduction",
        "Worked example / demonstration",
        "Interactive pause / knowledge check",
        "Summary and objective recap",
    ],
    review_criteria=[
        "The learning objective is stated explicitly near the start",
        "On-screen text (OST) beats are short enough to read in the shot duration",
        "At least one interactive pause prompt is present",
        "Narration avoids jargon the target learner wouldn't know",
        "The closing beat recaps the stated objective",
    ],
)
