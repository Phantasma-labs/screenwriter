# tests/test_formatters.py
from __future__ import annotations

from src.formatters.dual_column import (
    TABLE_HEADER,
    AVBeat,
    parse_av_beats,
    render_dual_column_table,
)


def test_parse_av_beats_valid_json():
    raw = (
        "Here is the script:\n"
        '[{"timecode": "0:00-0:03", "visual": "Logo reveal", "audio": "Upbeat sting"}]'
    )
    beats = parse_av_beats(raw)
    assert len(beats) == 1
    assert beats[0].timecode == "0:00-0:03"
    assert beats[0].visual == "Logo reveal"


def test_parse_av_beats_invalid_returns_empty():
    assert parse_av_beats("not json at all") == []


def test_render_dual_column_table_header_only_when_empty():
    assert render_dual_column_table([]) == TABLE_HEADER


def test_render_dual_column_table_includes_rows():
    beats = [AVBeat(timecode="0:00", visual="V1", audio="A1")]
    table = render_dual_column_table(beats)
    assert "| 0:00 | V1 | A1 |" in table
    assert table.startswith(TABLE_HEADER)
