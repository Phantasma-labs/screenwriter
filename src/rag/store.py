# src/rag/store.py
from __future__ import annotations

import uuid
from typing import Protocol

import chromadb

from src.parsers.chunker import Chunk


class EmbeddingsFn(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class EphemeralVectorStore:
    def __init__(self, embeddings: EmbeddingsFn, collection_name: str | None = None) -> None:
        self._client = chromadb.EphemeralClient()
        self._embeddings = embeddings
        self._collection = self._client.create_collection(
            name=collection_name or f"run-{uuid.uuid4().hex}"
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embeddings.embed_documents([c.text for c in chunks])
        self._collection.add(
            ids=[f"{c.source_file}:{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_file": c.source_file, "chunk_index": c.chunk_index} for c in chunks
            ],
        )

    def similarity_search(self, query: str, k: int) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        results = self._collection.query(query_embeddings=[query_vector], n_results=min(k, count))
        documents = results.get("documents") or [[]]
        return documents[0]


_ACTIVE_STORES: dict[str, EphemeralVectorStore] = {}


def register_store(run_id: str, store: EphemeralVectorStore) -> None:
    _ACTIVE_STORES[run_id] = store


def get_store(run_id: str) -> EphemeralVectorStore | None:
    return _ACTIVE_STORES.get(run_id)


def clear_store(run_id: str) -> None:
    _ACTIVE_STORES.pop(run_id, None)
