# src/graph/workflow.py
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings, load_settings
from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_overview,
    route_after_overview_discussion_wait,
    route_after_reviewer,
)
from src.graph.nodes.bible_reviewer import make_bible_reviewer_node
from src.graph.nodes.character_bible import make_character_bible_node
from src.graph.nodes.finalize import make_finalize_node
from src.graph.nodes.ingest import EmbeddingsFn, make_ingest_node
from src.graph.nodes.interviewer import interview_wait_node, make_interview_ask_node
from src.graph.nodes.location_bible import make_location_bible_node
from src.graph.nodes.outliner import make_outliner_node
from src.graph.nodes.overview import make_overview_node
from src.graph.nodes.overview_discussion import (
    make_overview_revise_node,
    overview_discussion_wait_node,
)
from src.graph.nodes.researcher import make_researcher_node
from src.graph.nodes.reviewer import make_reviewer_node
from src.graph.nodes.writer import make_writer_node
from src.graph.state import ScreenplayState
from src.tools.search import SearchResult, search


def build_workflow(
    enable_search: bool = False,
    llm: BaseChatModel | None = None,
    embeddings: EmbeddingsFn | None = None,
    search_fn: Callable[[str], list[SearchResult]] = search,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    settings = settings or load_settings()
    graph = StateGraph(ScreenplayState)

    graph.add_node("ingest", make_ingest_node(embeddings=embeddings))
    graph.add_node("interview_ask", make_interview_ask_node(llm=llm))
    graph.add_node("interview_wait", interview_wait_node)
    graph.add_node("researcher", make_researcher_node(enabled=enable_search, search_fn=search_fn))
    graph.add_node("outliner", make_outliner_node(llm=llm, rag_top_k=settings.rag_top_k))
    graph.add_node("writer", make_writer_node(llm=llm, rag_top_k=settings.rag_top_k))
    graph.add_node("reviewer", make_reviewer_node(llm=llm))
    graph.add_node("overview", make_overview_node(llm=llm))
    graph.add_node("overview_discussion_wait", overview_discussion_wait_node)
    graph.add_node("overview_revise", make_overview_revise_node(llm=llm))
    graph.add_node("character_bible", make_character_bible_node(llm=llm))
    graph.add_node("location_bible", make_location_bible_node(llm=llm))
    graph.add_node("bible_reviewer", make_bible_reviewer_node(llm=llm))
    graph.add_node("finalize", make_finalize_node(llm=llm))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "interview_ask")
    graph.add_conditional_edges(
        "interview_ask",
        route_after_interview_ask,
        {"interview_wait": "interview_wait", "researcher": "researcher"},
    )
    graph.add_conditional_edges(
        "interview_wait",
        route_after_interview_wait,
        {"interview_ask": "interview_ask", "researcher": "researcher"},
    )
    graph.add_edge("researcher", "outliner")
    graph.add_edge("outliner", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"writer": "writer", "overview": "overview"},
    )
    graph.add_conditional_edges(
        "overview",
        route_after_overview,
        {
            "overview_discussion_wait": "overview_discussion_wait",
            "character_bible": "character_bible",
        },
    )
    graph.add_conditional_edges(
        "overview_discussion_wait",
        route_after_overview_discussion_wait,
        {"overview_revise": "overview_revise", "character_bible": "character_bible"},
    )
    graph.add_edge("overview_revise", "overview_discussion_wait")
    graph.add_edge("character_bible", "location_bible")
    graph.add_edge("location_bible", "bible_reviewer")
    graph.add_conditional_edges(
        "bible_reviewer",
        route_after_bible_reviewer,
        {"character_bible": "character_bible", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=InMemorySaver())
