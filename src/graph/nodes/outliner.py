# src/graph/nodes/outliner.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.state import NodeUpdate, ScreenplayState
from src.rag.retriever import retrieve_context
from src.skills import get_skill


def make_outliner_node(
    llm: BaseChatModel | None = None,
    retrieve_fn: Callable[[str, str, int], list[str]] = retrieve_context,
    rag_top_k: int = 5,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.7)

    def outliner_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        transcript_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}" for t in state["interview_transcript"]
        )
        grounding_chunks = retrieve_fn(state["rag_run_id"], state["topic"], rag_top_k)
        grounding = "\n\n".join(grounding_chunks) or state["parsed_context"]
        messages = [
            SystemMessage(content=skill.system_prompt()),
            HumanMessage(
                content=(
                    f"{skill.outline_prompt()}\n\n"
                    f"Topic: {state['topic']}\n"
                    f"Interview notes:\n{transcript_text or '(none)'}\n\n"
                    f"Research notes:\n{state['research_notes'] or '(none)'}\n\n"
                    f"Source context:\n{grounding or '(none)'}"
                )
            ),
        ]
        response = llm.invoke(messages)
        return {"outline": str(response.content), "status": "outlined"}

    return outliner_node
