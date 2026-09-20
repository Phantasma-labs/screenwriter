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


def _with_warnings(context_text: str, warnings: list[str]) -> str:
    if not warnings:
        return context_text
    warning_block = "[INGEST WARNINGS]\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n"
    return warning_block + context_text


def make_ingest_node(
    embeddings: EmbeddingsFn | None = None,
    settings: Settings | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    settings = settings or load_settings()

    def ingest_node(state: ScreenplayState) -> NodeUpdate:
        if not state["file_paths"]:
            return {"parsed_context": "", "rag_indexed": False, "status": "ingested"}

        parsed = parse_context_files(state["file_paths"])
        warnings = list(parsed.warnings)

        if len(parsed.combined_text) >= settings.rag_min_chars_to_index:
            try:
                emb = embeddings or get_embeddings(settings)
                chunks = chunk_parsed_context(
                    parsed, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
                )
                run_id = f"run-{uuid.uuid4().hex}"
                store = EphemeralVectorStore(embeddings=emb, collection_name=run_id)
                store.add_chunks(chunks)
                register_store(run_id, store)
                return {
                    "parsed_context": _with_warnings(parsed.combined_text, warnings),
                    "status": "ingested",
                    "rag_run_id": run_id,
                    "rag_indexed": True,
                }
            except Exception as exc:  # noqa: BLE001 - any RAG failure falls back to direct context
                warnings.append(
                    f"RAG indexing failed ({exc}); passing context directly to the model instead."
                )

        return {
            "parsed_context": _with_warnings(parsed.combined_text, warnings),
            "status": "ingested",
            "rag_indexed": False,
        }

    return ingest_node
