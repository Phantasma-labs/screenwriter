# src/graph/nodes/researcher.py
from __future__ import annotations

from collections.abc import Callable

from src.graph.state import NodeUpdate, ScreenplayState
from src.tools.search import SearchResult, search


def make_researcher_node(
    enabled: bool, search_fn: Callable[[str], list[SearchResult]] = search
) -> Callable[[ScreenplayState], NodeUpdate]:
    def researcher_node(state: ScreenplayState) -> NodeUpdate:
        if not enabled:
            return {"research_notes": "", "status": "researched"}
        results = search_fn(state["topic"])
        notes = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)
        return {"research_notes": notes, "status": "researched"}

    return researcher_node
