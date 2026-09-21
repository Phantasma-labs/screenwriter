# src/skills/base.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    FOUNTAIN = "fountain"
    DUAL_COLUMN = "dual_column"


MASTER_SCENE_RULES = """MASTER SCENE FORMAT RULES (Fountain-compatible):
- Scene headings (sluglines) are ALL CAPS: "INT. LOCATION - DAY" / "EXT. LOCATION - NIGHT".
- Character cues are ALL CAPS on their own line directly above dialogue. Use
  extensions in parentheses when needed: (V.O.), (O.S.), (CONT'D).
- Parentheticals are short, lowercase, used sparingly - only when the manner
  of delivery isn't clear from the dialogue or action itself.
- Action/description blocks are written in present tense and kept to 4 lines
  or fewer per paragraph; break longer beats into multiple short paragraphs.
- Transitions are ALL CAPS and end in "TO:" (e.g. CUT TO:, DISSOLVE TO:) or
  are FADE IN: / FADE OUT., used sparingly.
- Subheaders: bare ALL-CAPS lines with no INT./EXT. (e.g. "HALLWAY",
  "FILBERT'S POV") shift position/POV within a scene without a full new
  slugline - use sparingly, only to break up complex or lengthy scenes.
- Montage / special sequences get their own ALL-CAPS label line immediately
  before the beats they cover (e.g. "INTERCUT PHONE CONVERSATION", "IN SLOW
  MOTION -") to make pacing/tone shifts explicit.
- Show, don't tell: action lines describe only what the camera can see."""

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


@dataclass(frozen=True)
class SkillProfile:
    name: str
    display_name: str
    output_format: OutputFormat
    persona: str
    outline_template: list[str]
    review_criteria: list[str]

    def system_prompt(self) -> str:
        return f"{self.persona}\n\n{MASTER_SCENE_RULES}"

    def review_system_prompt(self) -> str:
        checklist = "\n".join(f"- {c}" for c in self.review_criteria)
        return (
            f"You are a professional script doctor reviewing a {self.display_name} "
            f"script written by an AI writer.\n\nEvaluate strictly against this "
            f"checklist:\n{checklist}\n\nBe specific and actionable in your critique. "
            "A script scoring 8.0 or higher out of 10 passes."
        )

    def outline_prompt(self) -> str:
        beats = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(self.outline_template))
        return (
            f"Draft a beat sheet / treatment for a {self.display_name} following "
            f"this structure:\n{beats}"
        )
