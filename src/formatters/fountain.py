# src/formatters/fountain.py
from __future__ import annotations

import re

from src.formatters.dual_column import AVBeat

_SCENE_HEADING_RE = re.compile(r"^(INT|EXT|INT\./EXT|I/E)[\./ ]", re.IGNORECASE)


def render_fountain(draft: str) -> str:
    """Uppercases scene heading lines; leaves everything else as authored."""
    rendered_lines: list[str] = []
    for line in draft.splitlines():
        stripped = line.strip()
        if _SCENE_HEADING_RE.match(stripped):
            rendered_lines.append(stripped.upper())
        else:
            rendered_lines.append(line)
    return "\n".join(rendered_lines)


def validate_fountain(text: str) -> list[str]:
    """Best-effort format linter. Never raises; returns human-readable warnings."""
    warnings: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        if len(block) > 4:
            preview = " ".join(block)[:60]
            warnings.append(f"Action block has {len(block)} lines (max 4): {preview!r}")
        block.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_block()
            continue
        is_heading = bool(_SCENE_HEADING_RE.match(stripped))
        is_character_cue = stripped.isupper() and len(stripped.split()) <= 6
        if is_heading or is_character_cue:
            flush_block()
            continue
        block.append(stripped)
    flush_block()
    return warnings


def render_dual_column_as_fountain(beats: list[AVBeat]) -> str:
    """Maps Dual-Column A/V beats into Fountain conventions: action lines for
    IMAGE/DESCRIPTION, a parenthetical for TECHNICAL notes, and a NARRATOR
    character cue for NARRATION. SFX/OST/graphics notes the writer already
    wrote inline in image/description/narration text pass through as-is."""
    lines: list[str] = []
    for beat in beats:
        lines.append(f"[[{beat.timecode}]]")
        lines.append(beat.first_frame_image.strip())
        if beat.description.strip():
            lines.append(beat.description.strip())
        if beat.technical.strip():
            lines.append(f"({beat.technical.strip()})")
        lines.append("")
        lines.append("NARRATOR")
        lines.append(beat.narration.strip())
        lines.append("")
    return "\n".join(lines).rstrip("\n")
