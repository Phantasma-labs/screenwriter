# src/graph/nodes/finalize.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.dual_column import parse_av_beats, render_dual_column_table
from src.formatters.fountain import (
    render_dual_column_as_fountain,
    render_fountain,
    validate_fountain,
)
from src.graph.state import NodeUpdate, ScreenplayState
from src.rag.store import clear_store
from src.skills import get_skill
from src.skills.base import T2I_PROMPT_GUIDELINES, OutputFormat

_KEY_ART_SYSTEM_PROMPT = (
    "You write a single T2I key-art prompt for a screenplay's poster/key visual, "
    "capturing its tone and central image in one image.\n\n" + T2I_PROMPT_GUIDELINES
)


def make_finalize_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def finalize_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        key_art_response = llm.invoke(
            [
                SystemMessage(content=_KEY_ART_SYSTEM_PROMPT),
                HumanMessage(content=f"Screenplay draft:\n\n{state['draft']}"),
            ]
        )
        key_art_prompt = str(key_art_response.content)

        if skill.output_format == OutputFormat.FOUNTAIN:
            fountain_script = render_fountain(state["draft"])
            body = state["draft"]
        else:
            beats = parse_av_beats(state["draft"])
            if not beats:
                fountain_script = render_fountain(state["draft"])
                body = (
                    "**Note:** Could not parse the draft as structured A/V beats; "
                    "showing the raw draft below.\n\n" + state["draft"]
                )
            else:
                fountain_script = render_dual_column_as_fountain(beats)
                body = render_dual_column_table(beats)

        format_warnings = validate_fountain(fountain_script)
        warnings_section = (
            "\n## Formatting Warnings\n" + "\n".join(f"- {w}" for w in format_warnings) + "\n"
            if format_warnings
            else ""
        )
        key_art_box = f"**Key Art T2I Prompt:**\n```text\n{key_art_prompt}\n```\n"
        screenplay_markdown = f"{key_art_box}\n# Screenplay\n\n{body}{warnings_section}"

        if state.get("rag_indexed") and state.get("rag_run_id"):
            clear_store(state["rag_run_id"])

        return {
            "fountain_script": fountain_script,
            "screenplay_markdown": screenplay_markdown,
            "status": "finalized",
        }

    return finalize_node
