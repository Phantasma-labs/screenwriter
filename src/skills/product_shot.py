# src/skills/product_shot.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

PRODUCT_SHOT = SkillProfile(
    name="product_shot",
    display_name="Product Shot",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a spec-ad director specializing in sensorially rich product "
        "films. You script in Dual-Column format with macro-lens camera "
        "movements, deliberate lighting setups, and ambient foley/sound design "
        "notes for every beat."
    ),
    outline_template=[
        "Establishing atmosphere shot",
        "Macro hero reveal of the product",
        "Texture / material detail beats",
        "Motion / interaction beat",
        "Final hero frame + logo",
    ],
    review_criteria=[
        "Every visual cue specifies lens/framing (e.g. macro, slow motion, close up)",
        "Lighting is explicitly described for each major beat",
        "At least one foley/ambient sound design note appears in the audio column",
        "Pacing favors extended, lingering shots over quick cuts",
        "The product is the visual subject of the majority of beats",
    ],
)
