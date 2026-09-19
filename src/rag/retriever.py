# src/rag/retriever.py
from __future__ import annotations

from src.rag.store import get_store


def retrieve_context(run_id: str, query: str, k: int) -> list[str]:
    if not run_id:
        return []
    store = get_store(run_id)
    if store is None:
        return []
    return store.similarity_search(query, k)
