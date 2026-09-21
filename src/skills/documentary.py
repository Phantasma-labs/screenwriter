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
        "interview subject dialogue, and ambient sound notes. Vary shot duration "
        "realistically rather than a uniform cadence: brisk 2-6s cuts for "
        "archival/B-roll evidence, 3-8s for reconstruction/action beats, and "
        "occasional slower 10-12s holds only for contemplative or emotionally "
        "weighted moments."
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
        "Shot durations vary realistically across the piece (archival/B-roll "
        "cuts run 2-6s, reconstruction/action 3-8s, contemplative holds up to "
        "10-12s) rather than a uniform per-beat cadence",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
