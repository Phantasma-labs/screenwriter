# src/formatters/overview.py
from __future__ import annotations

from pydantic import BaseModel

from src.utils import extract_json_object


class OverviewEntry(BaseModel):
    story_description: str
    duration_estimate: str
    frame_format: str
    aspect_ratio: str
    camera: str
    lenses: str


FALLBACK_OVERVIEW = OverviewEntry(
    story_description="Overview response could not be parsed as valid JSON.",
    duration_estimate="Unknown",
    frame_format="Unknown",
    aspect_ratio="Unknown",
    camera="Unknown",
    lenses="Unknown",
)


def parse_overview(raw: str) -> OverviewEntry | None:
    return extract_json_object(raw, OverviewEntry)


def render_overview(entry: OverviewEntry) -> str:
    return (
        "# Overview\n\n"
        f"{entry.story_description}\n\n"
        "## Tech Specs\n\n"
        f"- **Duration:** {entry.duration_estimate}\n"
        f"- **Frame Format:** {entry.frame_format}\n"
        f"- **Aspect Ratio:** {entry.aspect_ratio}\n"
        f"- **Camera:** {entry.camera}\n"
        f"- **Lenses:** {entry.lenses}\n"
    )
