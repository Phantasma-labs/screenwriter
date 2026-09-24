# src/formatters/dual_column.py
from __future__ import annotations

from pydantic import BaseModel

from src.formatters.t2i import t2i_box
from src.utils import extract_json_array

SHOT_NUMBER_STEP = 10  # shots are numbered 010, 020, 030, ... in beat order

TABLE_HEADER = "| TIMECODE / BEAT | DESCRIPTION | NARRATION | TECHNICAL |\n|---|---|---|---|\n"


class AVBeat(BaseModel):
    timecode: str
    first_frame_image: str
    last_frame_image: str = ""
    i2v_prompt: str = ""
    description: str
    narration: str
    technical: str


def parse_av_beats(raw: str) -> list[AVBeat]:
    return extract_json_array(raw, AVBeat)


def _render_beat_card(beat: AVBeat, shot_number: int) -> str:
    sections = [f"## Shot {shot_number:03d} - Beat - {beat.timecode}\n"]
    sections.append(t2i_box("First Frame T2I Prompt", beat.first_frame_image))
    if beat.last_frame_image.strip():
        sections.append(t2i_box("Last Frame T2I Prompt", beat.last_frame_image))
    sections.append(t2i_box("I2V Prompt", beat.i2v_prompt))
    return "\n".join(sections)


def render_dual_column_table(beats: list[AVBeat]) -> str:
    if not beats:
        return TABLE_HEADER
    rows = "\n".join(
        f"| {b.timecode} | {b.description} | {b.narration} | {b.technical} |" for b in beats
    )
    table = TABLE_HEADER + rows + "\n"
    cards = "\n".join(
        _render_beat_card(b, shot_number=(i + 1) * SHOT_NUMBER_STEP) for i, b in enumerate(beats)
    )
    return table + "\n" + cards
