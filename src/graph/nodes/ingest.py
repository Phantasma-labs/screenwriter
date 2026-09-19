# src/graph/nodes/ingest.py
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Protocol

from src.config import Settings, get_embeddings, load_settings
from src.graph.state import NodeUpdate, ScreenplayState
from src.parsers.base import parse_context_files
from src.parsers.chunker import chunk_parsed_context
from src.rag.store import EphemeralVectorStore, register_store


class EmbeddingsFn(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def make_ingest_node(
    embeddings: EmbeddingsFn | None = None,
    settings: Settings | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    settings = settings or load_settings()

    def ingest_node(state: ScreenplayState) -> NodeUpdate:
        if not state["file_paths"]:
            return {"parsed_context": "", "rag_indexed": False, "status": "ingested"}

        parsed = parse_context_files(state["file_paths"])
        update: NodeUpdate = {"parsed_context": parsed.combined_text, "status": "ingested"}

        if len(parsed.combined_text) < settings.rag_min_chars_to_index:
            update["rag_indexed"] = False
            return update

        emb = embeddings or get_embeddings(settings)
        chunks = chunk_parsed_context(
            parsed, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
        )
        run_id = f"run-{uuid.uuid4().hex}"
        store = EphemeralVectorStore(embeddings=emb, collection_name=run_id)
        store.add_chunks(chunks)
        register_store(run_id, store)
        update["rag_run_id"] = run_id
        update["rag_indexed"] = True
        return update

    return ingest_node
