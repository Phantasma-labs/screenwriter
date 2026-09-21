# Dual-Column T2I/I2V Richness and Realistic Pacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make T2I prompts across the whole pipeline (character bible, location bible, and Dual-Column screenplay beats) full ~150-250 word cinematic paragraphs instead of one-line fragments; give every Dual-Column beat a First Frame T2I prompt, an optional Last Frame T2I prompt, and an I2V (image-to-video) motion prompt using FF/FFLF technique; fix the location bible picking up the character wardrobe/headshot studio backdrop as a fake "location"; and replace the uniform, too-long shot-duration pattern with realistic per-format pacing.

**Architecture:** `AVBeat` gains `last_frame_image`/`i2v_prompt` fields and renames `image` → `first_frame_image`; `Screenplay.md`'s rendering splits into a compact scannable table plus one expanded prompt "card" per beat (reusing a `t2i_box` helper extracted out of `bible.py` so both formatters share it); a new `I2V_PROMPT_GUIDELINES` constant joins the existing `T2I_PROMPT_GUIDELINES` (now with a concrete length target) in `writer.py`'s Dual-Column instruction; `location_bible.py` gets a second deterministic filter (studio-backdrop keywords) alongside its existing character-name cross-reference filter; and both `bible_reviewer.py` and each of the 5 Dual-Column skills' `review_criteria` get new enforcement bullets, since prompt-only guidance alone has already proven unreliable (the narrator/historian filter needed the same belt-and-suspenders treatment).

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, `FakeChatModel` test double (`tests/conftest.py`).

**Spec:** [docs/superpowers/specs/2026-09-21-dual-column-t2i-i2v-pacing-design.md](../specs/2026-09-21-dual-column-t2i-i2v-pacing-design.md)

## Global Constraints

- Every file starts with `from __future__ import annotations`; imports grouped stdlib / third-party / internal (ruff `I` rule).
- Ruff line length 100 (`[tool.ruff]` in `pyproject.toml`).
- Strict type annotations everywhere. No bare `Any`/unparameterized `dict`/`list` outside `NodeUpdate`.
- All tests run 100% offline — no live Ollama/Tavily/DuckDuckGo/Chroma calls. LLM calls are doubled with `FakeChatModel` from `tests/conftest.py`.
- `AVBeat.last_frame_image` and `AVBeat.i2v_prompt` get Pydantic string defaults of `""` (unlike the other required fields) so a beat isn't silently dropped by `extract_json_array` if a model omits one of these two keys entirely rather than writing an explicit empty string — matches this codebase's existing defensive-parsing posture (fallbacks over crashes) more closely than making them hard-required.
- No numeric/regex-based enforcement of the "150-250 words" target inside any node's code — enforcement is via the review loop's LLM judgment (`bible_reviewer_node` and each skill's `review_criteria`), consistent with how every other qualitative criterion in this codebase is already enforced.
- Per-skill pacing guidance is per-skill, not shared/centralized — matches the existing pattern where each skill fully owns its own `review_criteria` list (e.g. `"Every row has both a VISUAL and an AUDIO cue"` is already duplicated per skill rather than centralized).

---

### Task 1: Extract shared `t2i_box` helper

**Files:**
- Create: `src/formatters/t2i.py`
- Modify: `src/formatters/bible.py`
- Test: `tests/test_formatters.py`

**Interfaces:**
- Produces: `t2i_box(label: str, prompt: str) -> str` in `src/formatters/t2i.py`.
- Consumes (by `bible.py`, updated in this task): the new `t2i_box`, replacing its private `_t2i_box`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_formatters.py`, and add `from src.formatters.t2i import t2i_box` to its import block at the top:

```python
def test_t2i_box_formats_label_and_fenced_prompt():
    result = t2i_box("Headshot Prompt", "Create a portrait.")
    assert result == "**Headshot Prompt:**\n```text\nCreate a portrait.\n```\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatters.py::test_t2i_box_formats_label_and_fenced_prompt -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.formatters.t2i'`

- [ ] **Step 3: Write minimal implementation**

Create `src/formatters/t2i.py`:

```python
# src/formatters/t2i.py
from __future__ import annotations


def t2i_box(label: str, prompt: str) -> str:
    return f"**{label}:**\n```text\n{prompt}\n```\n"
```

In `src/formatters/bible.py`, replace the private helper with an import and remove the local definition:

```python
# src/formatters/bible.py
from __future__ import annotations

from pydantic import BaseModel

from src.formatters.t2i import t2i_box
from src.utils import extract_json_array


class CharacterBibleEntry(BaseModel):
    name: str
    role: str
    appearance: str
    personality: str
    voice: str
    backstory: str
    headshot_prompt: str
    contact_sheet_prompt: str
    wardrobe_prompt: str


class LocationBibleEntry(BaseModel):
    name: str
    description: str
    mood: str
    t2i_prompt: str


def parse_character_entries(raw: str) -> list[CharacterBibleEntry]:
    return extract_json_array(raw, CharacterBibleEntry)


def parse_location_entries(raw: str) -> list[LocationBibleEntry]:
    return extract_json_array(raw, LocationBibleEntry)


def render_character_bible(entries: list[CharacterBibleEntry]) -> str:
    if not entries:
        return "# Character Bible\n\nNo principal characters identified.\n"
    sections = ["# Character Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name} — {entry.role}\n")
        sections.append(f"**Appearance:** {entry.appearance}\n")
        sections.append(f"**Personality:** {entry.personality}\n")
        sections.append(f"**Voice:** {entry.voice}\n")
        sections.append(f"**Backstory:** {entry.backstory}\n")
        sections.append(t2i_box("Headshot Prompt", entry.headshot_prompt))
        sections.append(t2i_box("Contact Sheet Prompt", entry.contact_sheet_prompt))
        sections.append(t2i_box("Wardrobe & Accessories Prompt", entry.wardrobe_prompt))
    return "\n".join(sections)


def render_location_bible(entries: list[LocationBibleEntry]) -> str:
    if not entries:
        return "# Location Bible\n\nNo locations identified.\n"
    sections = ["# Location Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name}\n")
        sections.append(f"**Description:** {entry.description}\n")
        sections.append(f"**Mood:** {entry.mood}\n")
        sections.append(t2i_box("Location T2I Prompt", entry.t2i_prompt))
    return "\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatters.py -v`
Expected: All PASS, including the pre-existing `test_render_character_bible_includes_t2i_boxes` and `test_render_location_bible_includes_t2i_box` (unchanged behavior, just relocated implementation).

- [ ] **Step 5: Commit**

```bash
git add src/formatters/t2i.py src/formatters/bible.py tests/test_formatters.py
git commit -m "refactor: extract shared t2i_box helper out of bible.py"
```

---

### Task 2: `AVBeat` schema change and restructured rendering

**Files:**
- Modify: `src/formatters/dual_column.py`
- Modify: `src/formatters/fountain.py`
- Test: `tests/test_formatters.py`
- Test: `tests/test_graph_nodes.py`

**Interfaces:**
- Consumes: `t2i_box` (Task 1).
- Produces: `AVBeat` with fields `timecode: str, first_frame_image: str, last_frame_image: str = "", i2v_prompt: str = "", description: str, narration: str, technical: str` (renamed from `image`, two new fields with `""` defaults). `render_dual_column_table(beats: list[AVBeat]) -> str` now renders a 4-column table (`TIMECODE / BEAT | DESCRIPTION | NARRATION | TECHNICAL`) followed by one `## Beat <timecode>` card per beat with First Frame / (optional) Last Frame / I2V prompt boxes. `render_dual_column_as_fountain` (`fountain.py`) uses `beat.first_frame_image` instead of `beat.image`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_formatters.py`, replace `test_parse_av_beats_valid_json` and add a defaults test:

```python
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
```

Replace `test_render_dual_column_table_includes_rows` and add new card-rendering tests:

```python
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
```

Update `test_render_dual_column_as_fountain_produces_narrator_cues`:

```python
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
```

In `tests/test_graph_nodes.py`, update `test_finalize_node_dual_column_skill_produces_table_and_narrator_fountain`'s draft fixture and table-row assertion (the `IMAGE` column is gone from the table; the `first_frame_image` content now appears in the per-beat card instead):

```python
def test_finalize_node_dual_column_skill_produces_table_and_narrator_fountain():
    draft = (
        '[{"timecode": "0:00", "first_frame_image": "Logo reveal", '
        '"description": "Logo grows.", '
        '"narration": "Sting plays", "technical": "Slow zoom in"}]'
    )
    llm = FakeChatModel(responses=["Create a minimalist poster with a bold logo."])
    node = make_finalize_node(llm=llm)
    result = node(_state(skill="commercial", draft=draft))
    assert "| 0:00 | Logo grows. | Sting plays | Slow zoom in |" in result["screenplay_markdown"]
    assert "Logo reveal" in result["screenplay_markdown"]
    assert "NARRATOR" in result["fountain_script"]
    assert "Sting plays" in result["fountain_script"]
    assert "## Formatting Warnings" not in result["screenplay_markdown"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatters.py tests/test_graph_nodes.py -v`
Expected: FAIL — `AVBeat.__init__()` rejects `first_frame_image` as an unexpected keyword (field is still named `image`), and/or assertion mismatches on table content.

- [ ] **Step 3: Write minimal implementation**

Replace `src/formatters/dual_column.py` entirely:

```python
# src/formatters/dual_column.py
from __future__ import annotations

from pydantic import BaseModel

from src.formatters.t2i import t2i_box
from src.utils import extract_json_array

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


def _render_beat_card(beat: AVBeat) -> str:
    sections = [f"## Beat {beat.timecode}\n"]
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
    cards = "\n".join(_render_beat_card(b) for b in beats)
    return table + "\n" + cards
```

In `src/formatters/fountain.py`, change the one line in `render_dual_column_as_fountain` that reads `beat.image`:

```python
        lines.append(beat.first_frame_image.strip())
```

(This replaces the existing `lines.append(beat.image.strip())` line; nothing else in that function changes.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatters.py tests/test_graph_nodes.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/formatters/dual_column.py src/formatters/fountain.py tests/test_formatters.py tests/test_graph_nodes.py
git commit -m "feat: add last-frame and I2V prompts to AVBeat, restructure Screenplay.md rendering"
```

---

### Task 3: T2I length target and new I2V prompt guidelines

**Files:**
- Modify: `src/skills/base.py`
- Test: `tests/test_skills.py`

**Interfaces:**
- Produces: `T2I_PROMPT_GUIDELINES` (existing constant, gains a length-target bullet); `I2V_PROMPT_GUIDELINES: str` (new constant).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_skills.py`, and add `I2V_PROMPT_GUIDELINES` to the existing `from src.skills.base import ...` line:

```python
def test_t2i_guidelines_specifies_length_target():
    assert "150-250 words" in T2I_PROMPT_GUIDELINES


def test_i2v_guidelines_covers_ff_and_fflf_techniques():
    assert "FFLF" in I2V_PROMPT_GUIDELINES
    assert "FF (First-Frame-only)" in I2V_PROMPT_GUIDELINES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_skills.py -v`
Expected: FAIL — `ImportError: cannot import name 'I2V_PROMPT_GUIDELINES'` (and the length-target assertion fails once import is fixed).

- [ ] **Step 3: Write minimal implementation**

In `src/skills/base.py`, replace `T2I_PROMPT_GUIDELINES` and add `I2V_PROMPT_GUIDELINES` right after it:

```python
T2I_PROMPT_GUIDELINES = """T2I PROMPT GUIDELINES (target model: Nano Banana Pro):
- Formula: [Subject] + [Action] + [Location/context] + [Composition] + [Style].
- Length: a full paragraph, roughly 150-250 words (about 250 tokens) of concrete
  descriptive detail - or as much as the shot genuinely needs to fully specify
  subject, action, setting, composition, camera/lens, and lighting. Never a
  single short sentence or a bare fragment.
- Write full narrative sentences, never a bare keyword list.
- Use positive framing only - describe what IS in frame, never "no X".
- Open with a strong verb: "Create a...", "Generate an image of...".
- Use concrete materials/textures, not generic nouns (e.g. "navy blue tweed
  suit jacket", not "a suit").
- Specify camera, lens, and lighting explicitly (e.g. "shot on a Fujifilm
  camera, shallow depth of field f/1.8, three-point softbox lighting").
- Any on-screen text goes in quotes with font/style specified.
- Consistency without reference images: reuse one fixed canonical physical-
  description clause for a character (face, hair, build, signature
  color/prop) verbatim across all of that character's prompt boxes."""

I2V_PROMPT_GUIDELINES = """I2V PROMPT GUIDELINES (image-to-video motion direction):
- Write as a cinematographer directing motion between frames: camera movement
  (push in, pull out, pan, tilt, handheld drift, static), subject motion, and
  pacing - not a restatement of the still image's content.
- If last_frame_image is provided for this beat, write an FFLF (First-Frame-
  Last-Frame) prompt: describe the transformation FROM the first frame's
  composition TO the last frame's composition - what moves, changes, or
  reveals itself across the shot's duration.
- If last_frame_image is empty for this beat, write an FF (First-Frame-only)
  prompt: describe the motion that emanates from the single starting image
  alone - camera movement and/or subject action, without referencing an end
  state that wasn't specified.
- Match length and concreteness to the T2I prompt guidelines above - a full
  paragraph of specific direction, not a one-line note."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_skills.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skills/base.py tests/test_skills.py
git commit -m "feat: add T2I length target and I2V prompt guidelines"
```

---

### Task 4: Wire the new fields and guidelines into `writer.py`

**Files:**
- Modify: `src/graph/nodes/writer.py`
- Test: `tests/test_graph_nodes.py`

**Interfaces:**
- Consumes: `first_frame_image`/`last_frame_image`/`i2v_prompt` field names (Task 2), `T2I_PROMPT_GUIDELINES`/`I2V_PROMPT_GUIDELINES` (Task 3).
- Produces: the Dual-Column `format_instruction` string sent to the LLM now names the three fields correctly and includes both guideline blocks.

- [ ] **Step 1: Write the failing test**

Update `test_writer_node_uses_dual_column_instruction_for_commercial_skill` in `tests/test_graph_nodes.py`:

```python
def test_writer_node_uses_dual_column_instruction_for_commercial_skill():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(
        responses=[
            '[{"timecode": "0:00", "first_frame_image": "i", "description": "d", '
            '"narration": "n", "technical": "t"}]'
        ]
    )
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    node(_state(skill="commercial", outline="1. Hook"))
    human_content = str(captured_messages[0][-1].content)
    assert "JSON array" in human_content
    assert "first_frame_image" in human_content
    assert "150-250 words" in human_content
    assert "FFLF" in human_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py::test_writer_node_uses_dual_column_instruction_for_commercial_skill -v`
Expected: FAIL — `human_content` still says `"image"` (not `"first_frame_image"`) and contains neither `"150-250 words"` nor `"FFLF"`.

- [ ] **Step 3: Write minimal implementation**

In `src/graph/nodes/writer.py`, update the import line and the dual-column branch of `format_instruction`:

```python
from src.skills.base import I2V_PROMPT_GUIDELINES, OutputFormat, T2I_PROMPT_GUIDELINES
```

```python
        if skill.output_format == OutputFormat.FOUNTAIN:
            format_instruction = "Write the full script in Master Scene (Fountain) Format."
        else:
            format_instruction = (
                'Write the full script as a JSON array only: [{"timecode": str, '
                '"first_frame_image": str, "last_frame_image": str, "i2v_prompt": str, '
                '"description": str, "narration": str, "technical": str}, ...] '
                "- one object per beat. first_frame_image = the T2I prompt for the "
                "shot's starting frame; last_frame_image = a T2I prompt for the "
                "shot's ending frame, written only when the shot's visual state "
                'changes meaningfully within its duration (leave as an empty string ""'
                " otherwise); i2v_prompt = the image-to-video motion prompt (see I2V "
                "PROMPT GUIDELINES below); description = broader scene/action context "
                "beyond the T2I shots; narration = spoken/V.O. audio; technical = "
                "camera, lens, transition, or editing notes.\n\n"
                + T2I_PROMPT_GUIDELINES
                + "\n\n"
                + I2V_PROMPT_GUIDELINES
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/writer.py tests/test_graph_nodes.py
git commit -m "feat: wire T2I/I2V guidelines and new beat fields into the writer prompt"
```

---

### Task 5: Location bible excludes the wardrobe/headshot studio backdrop

**Files:**
- Modify: `src/graph/nodes/location_bible.py`
- Test: `tests/test_graph_nodes.py`

**Interfaces:**
- Consumes: `LocationBibleEntry` (`src/formatters/bible.py`).
- Produces: `_is_studio_backdrop(entry: LocationBibleEntry) -> bool` (new, private); `location_bible_node` drops entries matching it, in addition to its existing character-name filter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_nodes.py`, after the existing `test_location_bible_node_filters_character_names_case_insensitively` test:

```python
_LOCATION_WITH_BACKDROP_JSON = (
    '[{"name": "Office", "description": "A cramped detective office.", '
    '"mood": "Tense.", "t2i_prompt": "Create a wide shot of a cramped office."}, '
    '{"name": "Studio Backdrop", "description": "A plain, seamless white studio '
    'background used for reference shots.", '
    '"mood": "Neutral.", "t2i_prompt": "Create a seamless white studio backdrop."}]'
)


def test_location_bible_node_filters_out_studio_backdrop_entries():
    llm = FakeChatModel(responses=[_LOCATION_WITH_BACKDROP_JSON])
    node = make_location_bible_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert "Office" in result["location_bible"]
    assert "Studio Backdrop" not in result["location_bible"]


def test_location_bible_node_system_prompt_excludes_studio_backdrop():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=[_LOCATION_JSON])
    node = make_location_bible_node(llm=llm)
    node(_state(draft="draft text"))
    system_content = str(captured_messages[0][0].content).lower()
    assert "studio background" in system_content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py::test_location_bible_node_filters_out_studio_backdrop_entries tests/test_graph_nodes.py::test_location_bible_node_system_prompt_excludes_studio_backdrop -v`
Expected: FAIL — the "Studio Backdrop" entry currently survives (no filter yet), and the system prompt doesn't mention "studio background" yet.

- [ ] **Step 3: Write minimal implementation**

In `src/graph/nodes/location_bible.py`:

```python
# src/graph/nodes/location_bible.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.bible import LocationBibleEntry, parse_location_entries, render_location_bible
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills.base import T2I_PROMPT_GUIDELINES

_SYSTEM_PROMPT = (
    "You identify every key location in a script and write one production "
    "bible entry per location.\n\n"
    "Only include locations that actually appear on screen (host a scene via an "
    "INT./EXT. heading or an action beat) - not places that are merely "
    "mentioned in dialogue but never shown.\n\n"
    "A location entry describes a physical place or set - never a person. Do "
    "not write an entry for a character, historical figure, narrator, or "
    "interview subject, even if their name appears prominently in the draft; "
    "people belong in the character bible, not here.\n\n"
    "The plain, seamless white studio background used for character wardrobe "
    "and headshot reference shots is not a filming location - never write an "
    "entry for it.\n\n" + T2I_PROMPT_GUIDELINES
)

_RESPONSE_INSTRUCTION = (
    'Respond with a JSON array only: [{"name": str, "description": str, "mood": str, '
    '"t2i_prompt": str}, ...]'
)

_STUDIO_BACKDROP_KEYWORDS = (
    "seamless white",
    "white studio background",
    "white backdrop",
    "studio backdrop",
)


def _is_studio_backdrop(entry: LocationBibleEntry) -> bool:
    haystack = f"{entry.name} {entry.description}".lower()
    return any(keyword in haystack for keyword in _STUDIO_BACKDROP_KEYWORDS)


def make_location_bible_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def location_bible_node(state: ScreenplayState) -> NodeUpdate:
        feedback_note = (
            f"\n\nAddress this reviewer feedback:\n{state['bible_review_feedback']}"
            if state["bible_review_feedback"]
            else ""
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"{_RESPONSE_INSTRUCTION}{feedback_note}"
                )
            ),
        ]
        response = llm.invoke(messages)
        entries = parse_location_entries(str(response.content))
        character_names = {name.strip().casefold() for name in state["character_names"]}
        entries = [
            entry
            for entry in entries
            if entry.name.strip().casefold() not in character_names
            and not _is_studio_backdrop(entry)
        ]
        return {"location_bible": render_location_bible(entries)}

    return location_bible_node
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/location_bible.py tests/test_graph_nodes.py
git commit -m "fix: exclude the wardrobe/headshot studio backdrop from the location bible"
```

---

### Task 6: `bible_reviewer` enforces T2I prompt length

**Files:**
- Modify: `src/graph/nodes/bible_reviewer.py`
- Test: `tests/test_graph_nodes.py`

**Interfaces:** No new interfaces — this only strengthens the existing `_SYSTEM_PROMPT` string.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph_nodes.py`, near the other `bible_reviewer` tests:

```python
def test_bible_reviewer_node_system_prompt_requires_full_length_t2i_prompts():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(
        responses=[
            '{"score": 9.0, "passed": true, "critique": "Consistent.", "actionable_revisions": []}'
        ]
    )
    node = make_bible_reviewer_node(llm=llm)
    node(
        _state(
            draft="draft text",
            character_bible="# Character Bible\n",
            location_bible="# Location Bible\n",
            bible_revision_count=0,
        )
    )
    system_content = str(captured_messages[0][0].content).lower()
    assert "150-250 word" in system_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py::test_bible_reviewer_node_system_prompt_requires_full_length_t2i_prompts -v`
Expected: FAIL — current system prompt doesn't mention a length target.

- [ ] **Step 3: Write minimal implementation**

In `src/graph/nodes/bible_reviewer.py`, replace `_SYSTEM_PROMPT`:

```python
_SYSTEM_PROMPT = (
    "You are a production bible editor. Review the character and location bibles "
    "together against the screenplay draft they were derived from. Check: every "
    "principal character from the draft is represented, descriptions are consistent "
    "with how the character/location reads in the script, every entry has all "
    "required fields filled in with concrete (not generic) detail, and every T2I "
    "prompt (headshot, contact sheet, wardrobe, location) is a full ~150-250 word "
    "cinematic paragraph covering subject, action, setting, composition, "
    "camera/lens, and lighting - not a short sentence."
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph_nodes.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/bible_reviewer.py tests/test_graph_nodes.py
git commit -m "fix: bible_reviewer enforces full-length T2I prompts"
```

---

### Task 7: Per-skill realistic pacing and T2I/I2V richness criteria

**Files:**
- Modify: `src/skills/documentary.py`
- Modify: `src/skills/commercial.py`
- Modify: `src/skills/product_shot.py`
- Modify: `src/skills/learning_dev.py`
- Modify: `src/skills/infomedia.py`
- Test: `tests/test_skills.py`

**Interfaces:** No new interfaces — each skill's existing `persona`/`review_criteria` strings change. This is one task covering all five files: the same shape of edit repeated five times, reviewable as a unit.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_skills.py`:

```python
_DUAL_COLUMN_PACING = [
    ("documentary", "2-6s"),
    ("commercial", "1-4s"),
    ("product_shot", "3-8s"),
    ("learning_dev", "5-15s"),
    ("infomedia", "2-5s"),
]


@pytest.mark.parametrize("name,keyword", _DUAL_COLUMN_PACING)
def test_dual_column_skill_persona_mentions_realistic_pacing(name, keyword):
    skill = get_skill(name)
    assert keyword in skill.persona


@pytest.mark.parametrize(
    "name", ["documentary", "commercial", "product_shot", "learning_dev", "infomedia"]
)
def test_dual_column_skill_review_criteria_requires_t2i_i2v_richness(name):
    skill = get_skill(name)
    criteria_text = " ".join(skill.review_criteria)
    assert "150-250 word" in criteria_text
    assert "FFLF" in criteria_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_skills.py -v`
Expected: FAIL — none of the 5 skills' `persona`/`review_criteria` mention pacing or T2I/I2V richness yet.

- [ ] **Step 3: Write minimal implementation**

In `src/skills/documentary.py`, append to `persona` and add two `review_criteria` bullets:

```python
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
```

In `src/skills/commercial.py`:

```python
COMMERCIAL = SkillProfile(
    name="commercial",
    display_name="Commercial",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an award-winning commercial copywriter/director. You script "
        "high-tempo 15s/30s/60s spots in Dual-Column format: hook the viewer in "
        "the first 3 seconds, establish the problem, land an emotional beat, "
        "showcase the product, and close on a clear call to action (CTA). Cut "
        "fast and vary it: average 1-4s beats across the spot, with the "
        "CTA/brand lockup allowed to hold slightly longer at 2-4s."
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
        "Beat durations average 1-4s (matching real commercial cutting pace) "
        "and aren't a uniform cadence, with the CTA/lockup permitted to hold a "
        "bit longer",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
```

In `src/skills/product_shot.py` (note: this quantifies the existing "lingering shots" intent, it does not reverse it):

```python
PRODUCT_SHOT = SkillProfile(
    name="product_shot",
    display_name="Product Shot",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a spec-ad director specializing in sensorially rich product "
        "films. You script in Dual-Column format with macro-lens camera "
        "movements, deliberate lighting setups, and ambient foley/sound design "
        "notes for every beat. Hold most shots 3-8s and let hero/macro reveals "
        "breathe up to 10-12s - lingering pacing is intentional here, just not "
        "uniform."
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
        "Pacing favors extended, lingering shots (3-8s typical, hero/macro "
        "reveals up to 10-12s) over quick cuts, and isn't a uniform per-beat "
        "cadence",
        "The product is the visual subject of the majority of beats",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
```

In `src/skills/learning_dev.py`:

```python
LEARNING_DEV = SkillProfile(
    name="learning_dev",
    display_name="Learning & Development",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an instructional designer scripting a corporate training "
        "video. You script in Dual-Column format: the VISUAL column carries "
        "on-screen text (OST) and presenter direction, the AUDIO column carries "
        "narration written to a stated learning objective, with interactive "
        "pause prompts inserted where a learner should reflect or act. Hold "
        "shots long enough to read - typically 5-15s, driven by on-screen text "
        "length or demonstration pacing - varied rather than a fixed cadence."
    ),
    outline_template=[
        "Learning objective stated up front",
        "Concept introduction",
        "Worked example / demonstration",
        "Interactive pause / knowledge check",
        "Summary and objective recap",
    ],
    review_criteria=[
        "The learning objective is stated explicitly near the start",
        "On-screen text (OST) beats are short enough to read in the shot duration",
        "At least one interactive pause prompt is present",
        "Narration avoids jargon the target learner wouldn't know",
        "The closing beat recaps the stated objective",
        "Beat durations run 5-15s, sized to how long the on-screen "
        "text/demonstration in that beat actually takes to read or follow, not "
        "a uniform per-beat length",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
```

In `src/skills/infomedia.py`:

```python
INFOMEDIA = SkillProfile(
    name="infomedia",
    display_name="Infomedia / Explainer",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a motion-graphics explainer-video writer. You script in "
        "Dual-Column format using the Hook-Retain-Payoff structure: kinetic "
        "typography and infographic transitions in the VISUAL column, "
        "brisk narration paced to match the graphics in the AUDIO column. Cut "
        "briskly and vary it: 2-5s per graphic beat, faster still for kinetic "
        "typography moments."
    ),
    outline_template=[
        "Hook - the question or problem",
        "Retain - build understanding with kinetic graphics",
        "Payoff - the resolution or key takeaway",
        "Infographic summary beat",
        "Closing call to action",
    ],
    review_criteria=[
        "Follows Hook-Retain-Payoff structure recognizably",
        "Each visual beat names a specific motion-graphics treatment "
        "(kinetic type, icon animation, chart build, etc.)",
        "Narration pace notes (words per beat) are plausible for the stated beat duration",
        "Infographic/data beats are simple enough to parse on a single screen",
        "Closing beat delivers one clear takeaway, not several",
        "Beat durations run 2-5s (faster for kinetic typography) and vary "
        "rather than following a uniform cadence",
        "Every beat's First Frame T2I prompt (and Last Frame T2I prompt, when "
        "present) is a full ~150-250 word cinematic paragraph, and the I2V "
        "prompt correctly matches FF or FFLF technique to whether a last frame "
        "was written",
    ],
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_skills.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skills/documentary.py src/skills/commercial.py src/skills/product_shot.py src/skills/learning_dev.py src/skills/infomedia.py tests/test_skills.py
git commit -m "feat: add realistic per-format pacing and T2I/I2V richness criteria to Dual-Column skills"
```

---

### Task 8: Docs sync and full verification

**Files:**
- Modify: `claude.md`

**Interfaces:** None — documentation and verification only.

- [ ] **Step 1: Update the Dual-Column skills description in `claude.md`**

Find the `## Skills` section (the paragraph starting `Six \`SkillProfile\`s in \`src/skills/\`...`) and replace it:

Old:
```
## Skills
Six `SkillProfile`s in `src/skills/`: `short_film` (Master Scene Fountain
output) and `documentary`, `commercial`, `product_shot`, `learning_dev`,
`infomedia` (Dual-Column beat-table output: Timecode / Image / Description /
Narration / Technical per `AVBeat` in `src/formatters/dual_column.py` - Image
is the single literal T2I-promptable shot, Description is broader scene/action
context, Narration is spoken/V.O. audio, Technical is camera/lens/transition
notes; also mapped into Fountain conventions for `Script.md`). Each skill
defines its own system prompt, outline structure, and review checklist.
```

New:
```
## Skills
Six `SkillProfile`s in `src/skills/`: `short_film` (Master Scene Fountain
output) and `documentary`, `commercial`, `product_shot`, `learning_dev`,
`infomedia` (Dual-Column beat-table output: Timecode / First Frame Image /
Last Frame Image / I2V Prompt / Description / Narration / Technical per
`AVBeat` in `src/formatters/dual_column.py` - First Frame Image is the
literal T2I-promptable shot the beat opens on; Last Frame Image is an
optional second T2I prompt for the shot's ending frame, written only when
the shot's visual state changes meaningfully within its duration; I2V Prompt
is the image-to-video motion direction, written FFLF-style when a Last Frame
Image is present or FF-style when it isn't; Description is broader
scene/action context, Narration is spoken/V.O. audio, Technical is
camera/lens/transition notes; Screenplay.md renders a compact table plus a
per-beat card with the full T2I/I2V prompts, while Script.md maps only the
First Frame Image into Fountain conventions). Each skill defines its own
system prompt (including realistic per-format shot-pacing guidance),
outline structure, and review checklist.
```

- [ ] **Step 2: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -v --cov=src`
Expected: All tests PASS, including every test added/modified in Tasks 1-7.

- [ ] **Step 3: Run lint and format checks**

Run: `.venv/Scripts/python.exe -m ruff check src tests main.py app.py && .venv/Scripts/python.exe -m ruff format --check src tests main.py app.py`
Expected: No errors. If `ruff format --check` reports files needing
formatting, run `.venv/Scripts/python.exe -m ruff format src tests main.py app.py` and re-check.

- [ ] **Step 4: Commit**

```bash
git add claude.md
git commit -m "docs: document First/Last Frame + I2V beat fields in claude.md"
```
