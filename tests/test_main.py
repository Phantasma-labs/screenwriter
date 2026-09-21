# tests/test_main.py
from __future__ import annotations

from pathlib import Path

import pytest

from main import (
    is_finish_command,
    parse_args,
    print_outputs,
    resolve_output_dir,
    run_interactive,
    write_outputs,
)


class _FakeInterrupt:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeState:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values


class _FakeApp:
    def __init__(
        self, stream_batches: list[list[dict[str, object]]], final_values: dict[str, object]
    ) -> None:
        self._batches = list(stream_batches)
        self._final_values = final_values

    def stream(self, current_input, config, stream_mode="updates"):
        batch = self._batches.pop(0)
        yield from batch

    def get_state(self, config):
        return _FakeState(self._final_values)


@pytest.mark.parametrize("text", ["/finish", "/auto", "/autonomous", " /FINISH ", "/Auto"])
def test_is_finish_command_matches_known_phrases(text):
    assert is_finish_command(text) is True


@pytest.mark.parametrize("text", ["go ahead", "no", "", "finish"])
def test_is_finish_command_rejects_other_text(text):
    assert is_finish_command(text) is False


def test_parse_args_requires_topic():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["--topic", "A story"])
    assert args.skill == "short_film"
    assert args.files == []
    assert args.max_revisions == 2
    assert args.autonomous is False


def test_parse_args_rejects_unsupported_file_extension(tmp_path):
    bad_file = tmp_path / "image.png"
    bad_file.write_text("not really an image")
    with pytest.raises(SystemExit):
        parse_args(["--topic", "A story", "--files", str(bad_file)])


def test_parse_args_accepts_supported_extensions(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# notes")
    args = parse_args(["--topic", "A story", "--files", str(md_file)])
    assert args.files == [str(md_file)]


def test_resolve_output_dir_none_returns_none():
    assert resolve_output_dir(None) is None


def test_resolve_output_dir_returns_path_unchanged():
    assert resolve_output_dir("scripts/my_run") == Path("scripts/my_run")


def test_write_outputs_creates_five_fixed_named_files_in_directory(tmp_path):
    output_dir = tmp_path / "my_run"
    result = {
        "overview": "OVERVIEW TEXT",
        "fountain_script": "FOUNTAIN TEXT",
        "screenplay_markdown": "MARKDOWN TEXT",
        "character_bible": "CHAR BIBLE",
        "location_bible": "LOC BIBLE",
    }
    written = write_outputs(output_dir, result)
    assert (output_dir / "Overview.md").read_text() == "OVERVIEW TEXT"
    assert (output_dir / "Script.md").read_text() == "FOUNTAIN TEXT"
    assert (output_dir / "Screenplay.md").read_text() == "MARKDOWN TEXT"
    assert (output_dir / "CharacterBible.md").read_text() == "CHAR BIBLE"
    assert (output_dir / "LocationBible.md").read_text() == "LOC BIBLE"
    assert len(written) == 5


def test_print_outputs_includes_all_sections(capsys):
    result = {
        "overview": "OVERVIEW TEXT",
        "screenplay_markdown": "MARKDOWN TEXT",
        "fountain_script": "FOUNTAIN TEXT",
        "character_bible": "CHAR BIBLE",
        "location_bible": "LOC BIBLE",
    }
    print_outputs(result)
    out = capsys.readouterr().out
    assert "OVERVIEW TEXT" in out
    assert "MARKDOWN TEXT" in out
    assert "FOUNTAIN TEXT" in out
    assert "CHAR BIBLE" in out
    assert "LOC BIBLE" in out


def test_run_interactive_handles_one_interrupt_then_completes(monkeypatch, capsys):
    batches = [
        [{"ingest": {}}, {"__interrupt__": (_FakeInterrupt("What tone?"),)}],
        [{"outliner": {}}, {"writer": {}}, {"reviewer": {}}, {"finalize": {}}],
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})
    monkeypatch.setattr("builtins.input", lambda prompt="": "Melancholy")

    result = run_interactive(
        app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}}
    )

    assert result == {"status": "finalized"}
    assert "What tone?" in capsys.readouterr().out


def test_run_interactive_skips_empty_interrupt_payload(capsys):
    batches = [
        [
            {"ingest": {}},
            {"__interrupt__": ()},
            {"outliner": {}},
            {"writer": {}},
            {"reviewer": {}},
            {"finalize": {}},
        ]
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})

    result = run_interactive(
        app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}}
    )

    assert result == {"status": "finalized"}


def test_run_interactive_completes_immediately_when_no_interrupt(capsys):
    batches = [
        [{"ingest": {}}, {"outliner": {}}, {"writer": {}}, {"reviewer": {}}, {"finalize": {}}]
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})

    result = run_interactive(
        app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}}
    )

    assert result == {"status": "finalized"}


def test_run_interactive_handles_overview_discussion_interrupt(monkeypatch, capsys):
    batches = [
        [{"overview": {}}, {"__interrupt__": (_FakeInterrupt({"kind": "overview_discussion", "overview": "# Overview\n\nDraft."}),)}],
        [{"character_bible": {}}, {"finalize": {}}],
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})
    monkeypatch.setattr("builtins.input", lambda prompt="": "/finish")

    result = run_interactive(
        app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}}
    )

    assert result == {"status": "finalized"}
    out = capsys.readouterr().out
    assert "# Overview" in out
    assert "Draft." in out
