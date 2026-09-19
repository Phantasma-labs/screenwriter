# src/tools/search.py
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypedDict

from src.config import Settings, load_settings

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


TavilySearchFn = Callable[[str, str], list[SearchResult]]
DDGSSearchFn = Callable[[str], list[SearchResult]]


def _tavily_search(query: str, api_key: str) -> list[SearchResult]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in response.get("results", [])
    ]


def _ddgs_search(query: str) -> list[SearchResult]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=5))
    return [
        SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
        for r in raw_results
    ]


def search(
    query: str,
    settings: Settings | None = None,
    tavily_fn: TavilySearchFn = _tavily_search,
    ddgs_fn: DDGSSearchFn = _ddgs_search,
) -> list[SearchResult]:
    settings = settings or load_settings()
    if settings.tavily_api_key:
        try:
            return tavily_fn(query, settings.tavily_api_key)
        except Exception as exc:  # noqa: BLE001 - any Tavily failure falls back
            logger.warning("Tavily search failed (%s); falling back to DDGS.", exc)
    try:
        return ddgs_fn(query)
    except Exception as exc:  # noqa: BLE001 - both providers failed
        logger.warning("DDGS search failed (%s); returning no results.", exc)
        return []
