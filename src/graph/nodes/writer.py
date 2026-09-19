# src/graph/nodes/writer.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.state import NodeUpdate, ScreenplayState
from src.rag.retriever import retrieve_context
from src.skills import get_skill
from src.skills.base import OutputFormat


def make_writer_node(
    llm: BaseChatModel | None = None,
    retrieve_fn: Callable[[str, str, int], list[str]] = retrieve_context,
    rag_top_k: int = 5,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.8)

    def writer_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        grounding_chunks = retrieve_fn(state["rag_run_id"], state["outline"], rag_top_k)
        grounding = "\n\n".join(grounding_chunks) or state["parsed_context"]

        if skill.output_format == OutputFormat.FOUNTAIN:
            format_instruction = "Write the full script in Master Scene (Fountain) Format."
        else:
            format_instruction = (
                'Write the full script as a JSON array only: [{"timecode": str, '
                '"visual": str, "audio": str}, ...] - one object per beat.'
            )

        is_revision = bool(state["review_feedback"])
        revision_instruction = (
            f"\n\nThis is a revision. Address this feedback:\n{state['review_feedback']}"
            if is_revision
            else ""
        )
        messages = [
            SystemMessage(content=skill.system_prompt()),
            HumanMessage(
                content=(
                    f"Outline:\n{state['outline']}\n\n"
                    f"Source context:\n{grounding or '(none)'}\n\n"
                    f"{format_instruction}{revision_instruction}"
                )
            ),
        ]
        response = llm.invoke(messages)
        update: NodeUpdate = {"draft": str(response.content), "status": "drafted"}
        if is_revision:
            update["revision_count"] = state["revision_count"] + 1
        return update

    return writer_node
