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
        "pause prompts inserted where a learner should reflect or act. Hold "
        "shots long enough to read - typically 5-15s, driven by on-screen text "
        "length or demonstration pacing - varied rather than a fixed cadence."
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
        "Beat durations run 5-15s, sized to how long the on-screen "
        "text/demonstration in that beat actually takes to read or follow, not "
        "a uniform per-beat length",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
