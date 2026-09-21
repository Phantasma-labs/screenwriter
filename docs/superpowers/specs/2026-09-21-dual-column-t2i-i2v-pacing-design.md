# Dual-Column T2I Richness, First/Last Frame + I2V Prompts, and Realistic Pacing — Design

## Summary

Three related problems observed in generated output, all rooted in the same
class of failure (prompt-only guidance the model doesn't reliably follow, the
same lesson learned fixing the character-bible narrator filter):

1. **T2I prompts across the whole pipeline are too short.** Character bible,
   location bible, and Dual-Column screenplay beats all produce one-line T2I
   prompts (~25 tokens observed) instead of the rich, ~250-token cinematic
   paragraphs the tool is meant to generate. Root cause for the screenplay
   beats specifically: `writer.py`'s Dual-Column instruction never referenced
   the shared `T2I_PROMPT_GUIDELINES` at all. For character/location bibles,
   the guidance exists but has no concrete length target and nothing enforces
   it.
2. **Dual-Column beats only carry one T2I prompt** (the shot's starting
   image). There's no "last frame" prompt for shots whose visual state
   changes meaningfully, and no image-to-video (I2V) motion prompt at all —
   both needed for FF (first-frame-only) or FFLF (first-frame + last-frame)
   image-to-video generation workflows.
3. **Shot durations are uniformly patterned and too long** (~15s per beat,
   evenly spaced) because no skill has ever had realistic shot-duration
   guidance. Real professional pacing varies a lot by format (commercial cuts
   run 1-4s; product-shot beats deliberately linger 3-12s) and never sits at
   a fixed cadence.

Additionally, the location bible is independently picking up the neutral
"seamless white studio background" used only as a wardrobe/headshot
reference backdrop and listing it as if it were a real filming location.

## 1. `AVBeat` schema changes (`src/formatters/dual_column.py`)

```python
class AVBeat(BaseModel):
    timecode: str
    first_frame_image: str   # renamed from `image`
    last_frame_image: str    # NEW - "" when not needed for this beat
    i2v_prompt: str          # NEW - always populated
    description: str
    narration: str
    technical: str
```

- `first_frame_image` (rename of `image`): the T2I prompt for the shot's
  starting frame. Always populated.
- `last_frame_image`: a T2I prompt for the shot's ending frame. Populated
  only when the writer judges the shot's visual state changes meaningfully
  within its duration (camera reveals something new, an action completes, a
  transition happens) — left as `""` for shots that are essentially static
  within their own duration (e.g. a still archival photo with just a slow
  pan/zoom). This is an LLM judgment call per beat, not a rule tied to shot
  category — deliberately, since it varies shot to shot even within one
  category.
- `i2v_prompt`: always populated. Written in FFLF style (describes the
  transformation from the first frame's composition to the last frame's)
  when `last_frame_image` is non-empty; written in FF style (describes
  motion emanating from the single starting image alone) when it's empty.

This is a breaking rename of `image` → `first_frame_image` — acceptable
since nothing has shipped yet. Touches: `writer.py`'s JSON response
instruction, `dual_column.py`'s model/render function, `fountain.py`'s
`render_dual_column_as_fountain` (uses `beat.image` as the Fountain action
line → `beat.first_frame_image`), and every test fixture that constructs
AVBeat JSON (`tests/test_formatters.py`, `tests/test_graph_nodes.py`).
`finalize.py` doesn't reference the field directly (only parses/renders via
the formatter functions), so no change needed there beyond what the
renamed/extended model already provides.

## 2. T2I prompt richness (`src/skills/base.py`)

Add a concrete, enforceable length target to the shared `T2I_PROMPT_GUIDELINES`
constant (used by character bible, location bible, key art, and now the
Dual-Column beats):

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
```

Word count (not raw token count) is the anchor phrase, since that's what an
LLM can actually self-regulate against; "about 250 tokens" is kept as a
parenthetical for the user's own framing.

This alone is prompt-only guidance, which — per the narrator-filter lesson —
isn't reliably followed. Real enforcement comes from two review loops that
already exist:

- **`bible_reviewer_node`** (`src/graph/nodes/bible_reviewer.py`): add to its
  system prompt: "...and every T2I prompt (headshot, contact sheet, wardrobe,
  location) is a full ~150-250 word cinematic paragraph covering subject,
  action, setting, composition, camera/lens, and lighting - not a short
  sentence." This is the actor-critic loop that already re-invokes
  `character_bible`/`location_bible` on a failing score, so an under-length
  prompt now becomes a real, actionable revision trigger instead of a
  silent miss.
- **Each Dual-Column skill's `review_criteria`** (see section 4): a bullet
  requiring First/Last Frame T2I prompts to meet the same length bar, and
  the I2V prompt to correctly match FF/FFLF technique to whether a last
  frame was written. This plugs into the existing `writer`/`reviewer` loop
  the same way.

## 3. I2V prompt guidance (`src/skills/base.py`, used by `writer.py`)

A new sibling constant, used only by the Dual-Column `format_instruction`
(character/location bible and key art don't have motion to describe):

```python
I2V_PROMPT_GUIDELINES = """I2V PROMPT GUIDELINES (image-to-video, cinematographer's motion direction):
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

## 4. `writer.py`'s Dual-Column instruction

```python
format_instruction = (
    'Write the full script as a JSON array only: [{"timecode": str, '
    '"first_frame_image": str, "last_frame_image": str, "i2v_prompt": str, '
    '"description": str, "narration": str, "technical": str}, ...] '
    "- one object per beat. first_frame_image = the T2I prompt for the shot's "
    "starting frame; last_frame_image = a T2I prompt for the shot's ending "
    'frame, written only when the shot\'s visual state changes meaningfully '
    'within its duration (leave as an empty string "" otherwise); i2v_prompt '
    "= the image-to-video motion prompt (see I2V PROMPT GUIDELINES below); "
    "description = broader scene/action context beyond the T2I shots; "
    "narration = spoken/V.O. audio; technical = camera, lens, transition, or "
    "editing notes.\n\n" + T2I_PROMPT_GUIDELINES + "\n\n" + I2V_PROMPT_GUIDELINES
)
```

## 5. Screenplay.md layout (`render_dual_column_table`)

Table narrows to the scannable columns; full prompts move to per-beat cards
below it, mirroring how `bible.py` already presents long T2I prompts (label
+ fenced code block) rather than cramming them into a table cell:

```
| TIMECODE / BEAT | DESCRIPTION | NARRATION | TECHNICAL |
|---|---|---|---|
| 00:00 | ... | ... | ... |
...

## Beat 00:00

**First Frame T2I Prompt:**
```text
...
```

**Last Frame T2I Prompt:**   (only when beat.last_frame_image is non-empty)
```text
...
```

**I2V Prompt:**
```text
...
```
```

The `**Label:**\n\`\`\`text\n...\n\`\`\`\n` helper (`_t2i_box`, currently
private to `bible.py`) is extracted to a new shared module
`src/formatters/t2i.py` as a public `t2i_box(label, prompt) -> str`, and both
`bible.py` and `dual_column.py` import it — two real call sites doing the
identical thing is enough to justify sharing it, not premature abstraction.

`render_dual_column_as_fountain` (Script.md, the industry-standard shooting
script view) is **not** restructured — it keeps using
`beat.first_frame_image.strip()` as the action line (same role `beat.image`
played before the rename) and deliberately does not surface
`last_frame_image`/`i2v_prompt`; those belong to the T2I-prompt view
(Screenplay.md), not the script-reading view.

## 6. Location bible: exclude the wardrobe/headshot backdrop

`location_bible_node` already cross-references `state["character_names"]`
(from the earlier narrator/historian fix) to drop entries that are really
people. Add a second, independent filter for entries that are really the
neutral studio backdrop character_bible's wardrobe/headshot prompts
describe ("...against a plain, seamless white studio background..."):

```python
_STUDIO_BACKDROP_KEYWORDS = (
    "seamless white",
    "white studio background",
    "white backdrop",
    "studio backdrop",
)

def _is_studio_backdrop(entry: LocationBibleEntry) -> bool:
    haystack = f"{entry.name} {entry.description}".lower()
    return any(keyword in haystack for keyword in _STUDIO_BACKDROP_KEYWORDS)
```

Applied alongside the existing character-name filter:

```python
entries = [
    entry
    for entry in entries
    if entry.name.strip().casefold() not in character_names
    and not _is_studio_backdrop(entry)
]
```

Plus a matching sentence added to `location_bible.py`'s system prompt: the
studio backdrop used for character wardrobe/headshot references is not a
filming location and must never get its own entry.

## 7. Per-skill pacing (`persona` + `review_criteria` in each of the 5 skill files)

Per-skill (not shared), since realistic pacing genuinely differs by format —
confirmed by `product_shot.py` already explicitly wanting *slower* pacing
("favors extended, lingering shots") than the others. Each skill gets a
`persona` sentence (proactive) and matching `review_criteria` bullets
(enforcement, closing the writer/reviewer loop the same way length
enforcement does):

| Skill | Realistic range | Notes |
|---|---|---|
| `documentary` | archival/B-roll 2-6s, reconstruction/action 3-8s, contemplative holds up to 10-12s | mixed, no uniform cadence |
| `commercial` | 1-4s average | CTA/brand lockup may hold 2-4s |
| `product_shot` | 3-8s typical, hero/macro reveals up to 10-12s | quantifies the *existing* "lingering shots" intent, doesn't reverse it |
| `learning_dev` | 5-15s | sized to how long the on-screen text/demonstration in that beat takes to read/follow |
| `infomedia` | 2-5s | faster for kinetic typography moments |

Each skill's `review_criteria` also gets one shared-wording bullet: "Every
beat's First Frame T2I prompt (and Last Frame T2I prompt, when present) is a
full ~150-250 word cinematic paragraph, and the I2V prompt correctly matches
FF or FFLF technique to whether a last frame was written."

## 8. Docs

`claude.md`'s Skills section (the `Dual-Column beat-table output` sentence)
updated to describe `Timecode / First Frame Image / Last Frame Image / I2V
Prompt / Description / Narration / Technical` and each field's purpose,
replacing the current `Image is the single literal T2I-promptable shot`
description.

## 9. Testing

- `tests/test_formatters.py`, `tests/test_graph_nodes.py`: update every AVBeat
  JSON fixture (`"image"` → `"first_frame_image"`, add `"last_frame_image"`,
  `"i2v_prompt"`) and add coverage for: the new fields round-tripping through
  `parse_av_beats`; `render_dual_column_table`'s narrowed table + per-beat
  cards, including the conditional Last Frame card; the extracted `t2i_box`
  helper (wherever `bible.py`'s existing box-rendering gets its coverage,
  plus a `dual_column.py`-side test using the shared import).
- `tests/test_graph_nodes.py`: `make_writer_node`'s Dual-Column
  `format_instruction` includes `T2I_PROMPT_GUIDELINES`'s length target and
  `I2V_PROMPT_GUIDELINES`'s FF/FFLF instruction (system/human-message content
  assertions, matching the existing pattern for other nodes' prompt-content
  tests); `location_bible_node`'s new studio-backdrop filter (a fixture entry
  named/described as a seamless white backdrop gets dropped, alongside the
  existing character-name-match test).
- `tests/test_skills.py`: each of the 5 skills' `persona`/`review_criteria`
  mentions its pacing range and the T2I/I2V richness bullet (content
  assertions, not full-generation tests — matches how skill profiles are
  already tested).
- `bible_reviewer.py`'s strengthened system prompt: a content assertion test
  (new, since this node currently has none checking prompt content — only
  behavior tests).
- Full offline suite stays 100% offline throughout (`FakeChatModel`, no live
  calls) — none of this touches the RAG/search/live-LLM boundaries.

## Out of scope

- No change to `render_dual_column_as_fountain`/Script.md's structure beyond
  the field rename.
- No change to `character_bible.py`'s narrator/historian filter (already
  fixed and out of scope here) beyond it remaining the source of
  `character_names` that `location_bible.py`'s filter consumes.
- No centralized/shared review-criteria mechanism across the 5 skills — each
  continues to fully own its own `review_criteria` list, consistent with the
  existing pattern (e.g. "Every row has both a VISUAL and an AUDIO cue" is
  already duplicated per skill rather than centralized).
- No numeric/regex-based enforcement of the 150-250 word target in code (no
  word-count assertion inside a node) — enforcement is via the review loop's
  LLM judgment, consistent with how every other qualitative criterion in
  `review_criteria` is already enforced in this codebase.
