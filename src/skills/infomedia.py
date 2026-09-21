# src/skills/infomedia.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

INFOMEDIA = SkillProfile(
    name="infomedia",
    display_name="Infomedia / Explainer",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a motion-graphics explainer-video writer. You script in "
        "Dual-Column format using the Hook-Retain-Payoff structure: kinetic "
        "typography and infographic transitions in the VISUAL column, "
        "brisk narration paced to match the graphics in the AUDIO column. Cut "
        "briskly and vary it: 2-5s per graphic beat, faster still for kinetic "
        "typography moments."
    ),
    outline_template=[
        "Hook - the question or problem",
        "Retain - build understanding with kinetic graphics",
        "Payoff - the resolution or key takeaway",
        "Infographic summary beat",
        "Closing call to action",
    ],
    review_criteria=[
        "Follows Hook-Retain-Payoff structure recognizably",
        "Each visual beat names a specific motion-graphics treatment "
        "(kinetic type, icon animation, chart build, etc.)",
        "Narration pace notes (words per beat) are plausible for the stated beat duration",
        "Infographic/data beats are simple enough to parse on a single screen",
        "Closing beat delivers one clear takeaway, not several",
        "Beat durations run 2-5s (faster for kinetic typography) and vary "
        "rather than following a uniform cadence",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
