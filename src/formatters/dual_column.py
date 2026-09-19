# src/formatters/dual_column.py
from __future__ import annotations

from pydantic import BaseModel

from src.utils import extract_json_array

TABLE_HEADER = (
    "| TIMECODE / BEAT | VISUAL & CAMERA (VIDEO) | AUDIO, DIALOGUE & SFX |\n|---|---|---|\n"
)


class AVBeat(BaseModel):
    timecode: str
    visual: str
    audio: str


def parse_av_beats(raw: str) -> list[AVBeat]:
    return extract_json_array(raw, AVBeat)


def render_dual_column_table(beats: list[AVBeat]) -> str:
    if not beats:
        return TABLE_HEADER
    rows = "\n".join(f"| {b.timecode} | {b.visual} | {b.audio} |" for b in beats)
    return TABLE_HEADER + rows + "\n"
