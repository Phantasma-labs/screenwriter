# tests/test_formatters.py
from __future__ import annotations

from src.formatters.dual_column import (
    TABLE_HEADER,
    AVBeat,
    parse_av_beats,
    render_dual_column_table,
)
from src.formatters.fountain import (
    render_dual_column_as_fountain,
    render_fountain,
    validate_fountain,
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


def test_render_fountain_uppercases_lowercase_scene_heading():
    draft = "int. kitchen - day\n\nShe walks in and sits down.\n"
    result = render_fountain(draft)
    assert result.splitlines()[0] == "INT. KITCHEN - DAY"


def test_render_fountain_leaves_action_lines_unchanged():
    draft = "INT. KITCHEN - DAY\n\nShe walks in.\n"
    result = render_fountain(draft)
    assert "She walks in." in result


def test_validate_fountain_flags_long_action_block():
    text = "INT. KITCHEN - DAY\n\nOne.\nTwo.\nThree.\nFour.\nFive.\n"
    warnings = validate_fountain(text)
    assert len(warnings) == 1
    assert "5 lines" in warnings[0]


def test_validate_fountain_allows_short_action_block():
    text = "INT. KITCHEN - DAY\n\nShe enters.\nShe sits.\n"
    assert validate_fountain(text) == []


def test_validate_fountain_resets_after_character_cue():
    text = "INT. KITCHEN - DAY\n\nOne.\nTwo.\nThree.\nFour.\n\nJANE\nHello there.\n"
    assert validate_fountain(text) == []


def test_render_dual_column_as_fountain_produces_narrator_cues():
    beats = [
        AVBeat(timecode="0:00-0:03", visual="Logo reveal on black.", audio="Upbeat sting plays.")
    ]
    result = render_dual_column_as_fountain(beats)
    assert "[[0:00-0:03]]" in result
    assert "Logo reveal on black." in result
    assert "NARRATOR" in result
    assert "Upbeat sting plays." in result
