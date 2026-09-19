# tests/test_workflow.py
from __future__ import annotations

from src.graph.workflow import build_workflow


def test_build_workflow_compiles_with_all_expected_nodes():
    app = build_workflow(enable_search=False)
    node_names = set(app.get_graph().nodes.keys())
    for expected in [
        "ingest",
        "interview_ask",
        "interview_wait",
        "researcher",
        "outliner",
        "writer",
        "reviewer",
        "character_bible",
        "location_bible",
        "bible_reviewer",
        "finalize",
    ]:
        assert expected in node_names
