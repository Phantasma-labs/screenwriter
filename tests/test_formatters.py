# tests/test_formatters.py
from __future__ import annotations

from src.formatters.bible import (
    CharacterBibleEntry,
    LocationBibleEntry,
    parse_character_entries,
    parse_location_entries,
    render_character_bible,
    render_location_bible,
)
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
from src.formatters.t2i import t2i_box


def test_parse_av_beats_valid_json():
    raw = (
        "Here is the script:\n"
        '[{"timecode": "0:00-0:03", "first_frame_image": "Logo reveal", '
        '"last_frame_image": "Logo fully formed, static.", '
        '"i2v_prompt": "Logo grows from a single point into full reveal.", '
        '"description": "Logo grows.", "narration": "Upbeat sting", "technical": "Slow zoom in"}]'
    )
    beats = parse_av_beats(raw)
    assert len(beats) == 1
    assert beats[0].timecode == "0:00-0:03"
    assert beats[0].first_frame_image == "Logo reveal"
    assert beats[0].last_frame_image == "Logo fully formed, static."
    assert beats[0].i2v_prompt == "Logo grows from a single point into full reveal."
    assert beats[0].description == "Logo grows."
    assert beats[0].narration == "Upbeat sting"
    assert beats[0].technical == "Slow zoom in"


def test_parse_av_beats_defaults_last_frame_and_i2v_when_omitted():
    raw = (
        '[{"timecode": "0:00", "first_frame_image": "A shot.", '
        '"description": "D", "narration": "N", "technical": "T"}]'
    )
    beats = parse_av_beats(raw)
    assert len(beats) == 1
    assert beats[0].last_frame_image == ""
    assert beats[0].i2v_prompt == ""


def test_parse_av_beats_invalid_returns_empty():
    assert parse_av_beats("not json at all") == []


def test_render_dual_column_table_header_only_when_empty():
    assert render_dual_column_table([]) == TABLE_HEADER


def test_render_dual_column_table_includes_rows():
    beats = [
        AVBeat(
            timecode="0:00",
            first_frame_image="I1",
            description="D1",
            narration="N1",
            technical="T1",
        )
    ]
    table = render_dual_column_table(beats)
    assert "| 0:00 | D1 | N1 | T1 |" in table
    assert table.startswith(TABLE_HEADER)


def test_render_dual_column_table_includes_first_frame_and_i2v_cards():
    beats = [
        AVBeat(
            timecode="0:00",
            first_frame_image="A detailed first frame prompt.",
            description="D1",
            narration="N1",
            technical="T1",
        )
    ]
    table = render_dual_column_table(beats)
    assert "## Beat 0:00" in table
    assert "**First Frame T2I Prompt:**" in table
    assert "A detailed first frame prompt." in table
    assert "**I2V Prompt:**" in table


def test_render_dual_column_table_omits_last_frame_card_when_empty():
    beats = [
        AVBeat(
            timecode="0:00",
            first_frame_image="I1",
            description="D1",
            narration="N1",
            technical="T1",
        )
    ]
    table = render_dual_column_table(beats)
    assert "**Last Frame T2I Prompt:**" not in table


def test_render_dual_column_table_includes_last_frame_card_when_present():
    beats = [
        AVBeat(
            timecode="0:00",
            first_frame_image="I1",
            last_frame_image="The end state of the shot.",
            description="D1",
            narration="N1",
            technical="T1",
        )
    ]
    table = render_dual_column_table(beats)
    assert "**Last Frame T2I Prompt:**" in table
    assert "The end state of the shot." in table


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
        AVBeat(
            timecode="0:00-0:03",
            first_frame_image="Logo reveal on black.",
            description="Logo grows to fill frame.",
            narration="Upbeat sting plays.",
            technical="Slow zoom in, 50mm.",
        )
    ]
    result = render_dual_column_as_fountain(beats)
    assert "[[0:00-0:03]]" in result
    assert "Logo reveal on black." in result
    assert "Logo grows to fill frame." in result
    assert "Slow zoom in, 50mm." in result
    assert "NARRATOR" in result
    assert "Upbeat sting plays." in result


def _character() -> CharacterBibleEntry:
    return CharacterBibleEntry(
        name="Jane Voss",
        role="Protagonist",
        appearance="Sharp business attire, tired eyes, close-cropped dark hair.",
        personality="Relentless, dryly funny under pressure.",
        voice="Clipped, impatient, precise.",
        backstory="Former detective turned voicemail-service owner.",
        headshot_prompt="Create a headshot of Jane Voss, a woman with tired eyes.",
        contact_sheet_prompt="Create a 3x2 contact sheet of Jane Voss in six expressions.",
        wardrobe_prompt="Create a full-body wardrobe shot of Jane Voss in sharp business attire.",
    )


def _location() -> LocationBibleEntry:
    return LocationBibleEntry(
        name="Voss Voicemail Office",
        description="A cramped, fluorescent-lit office stacked with old answering machines.",
        mood="Tense and claustrophobic.",
        t2i_prompt="Create a wide shot of a cramped, fluorescent-lit detective office at night.",
    )


def test_parse_character_entries_valid_json():
    raw = f"[{_character().model_dump_json()}]"
    entries = parse_character_entries(raw)
    assert len(entries) == 1
    assert entries[0].name == "Jane Voss"


def test_parse_character_entries_invalid_returns_empty():
    assert parse_character_entries("not json") == []


def test_render_character_bible_empty_list():
    expected = "# Character Bible\n\nNo principal characters identified.\n"
    assert render_character_bible([]) == expected


def test_render_character_bible_includes_t2i_boxes():
    rendered = render_character_bible([_character()])
    assert "Jane Voss" in rendered
    assert "**Headshot Prompt:**" in rendered
    assert "```text" in rendered
    assert "Create a headshot of Jane Voss" in rendered
    assert "**Contact Sheet Prompt:**" in rendered
    assert "**Wardrobe & Accessories Prompt:**" in rendered


def test_parse_location_entries_valid_json():
    raw = f"[{_location().model_dump_json()}]"
    entries = parse_location_entries(raw)
    assert len(entries) == 1
    assert entries[0].name == "Voss Voicemail Office"


def test_render_location_bible_empty_list():
    assert render_location_bible([]) == "# Location Bible\n\nNo locations identified.\n"


def test_render_location_bible_includes_t2i_box():
    rendered = render_location_bible([_location()])
    assert "Voss Voicemail Office" in rendered
    assert "**Location T2I Prompt:**" in rendered
    assert "```text" in rendered
    assert "cramped, fluorescent-lit detective office" in rendered


def test_t2i_box_formats_label_and_fenced_prompt():
    result = t2i_box("Headshot Prompt", "Create a portrait.")
    assert result == "**Headshot Prompt:**\n```text\nCreate a portrait.\n```\n"
