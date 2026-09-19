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
- Show, don't tell: action lines describe only what the camera can see."""

T2I_PROMPT_GUIDELINES = """T2I PROMPT GUIDELINES (target model: Nano Banana Pro):
- Formula: [Subject] + [Action] + [Location/context] + [Composition] + [Style].
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
