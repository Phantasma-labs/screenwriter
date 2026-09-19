# src/skills/commercial.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

COMMERCIAL = SkillProfile(
    name="commercial",
    display_name="Commercial",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an award-winning commercial copywriter/director. You script "
        "high-tempo 15s/30s/60s spots in Dual-Column format: hook the viewer in "
        "the first 3 seconds, establish the problem, land an emotional beat, "
        "showcase the product, and close on a clear call to action (CTA)."
    ),
    outline_template=[
        "Hook (0-3s)",
        "Problem / tension",
        "Emotional turn / brand promise",
        "Product hero shot",
        "Call to action + brand lockup",
    ],
    review_criteria=[
        "The hook lands within the first 3 seconds of screen time",
        "Exactly one clear call to action appears near the end",
        "Product placement is specific (shot type, timing) not generic",
        "Runtime implied by the beats matches the target spot length",
        "Emotional beat is earned, not just stated in VO",
    ],
)
