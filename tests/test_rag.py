# tests/test_rag.py
from __future__ import annotations

from src.parsers.chunker import Chunk
from src.rag.retriever import retrieve_context
from src.rag.store import EphemeralVectorStore, clear_store, get_store, register_store


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        first = ord(text[0]) if text else 0.0
        return [float(len(text)), float(vowels), float(first)]


def test_add_chunks_and_similarity_search_returns_closest_text():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-1")
    chunks = [
        Chunk(text="The quick brown fox", source_file="a.md", chunk_index=0),
        Chunk(text="Quantum physics lecture notes", source_file="b.md", chunk_index=0),
    ]
    store.add_chunks(chunks)
    results = store.similarity_search("The quick brown fox", k=1)
    assert results == ["The quick brown fox"]


def test_similarity_search_empty_collection_returns_empty_list():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-2")
    assert store.similarity_search("anything", k=3) == []


def test_add_chunks_noop_on_empty_list():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-3")
    store.add_chunks([])
    assert store.similarity_search("x", k=1) == []


def test_register_and_get_store_roundtrip():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-4")
    register_store("run-123", store)
    try:
        assert get_store("run-123") is store
    finally:
        clear_store("run-123")
    assert get_store("run-123") is None


def test_get_store_unknown_run_id_returns_none():
    assert get_store("no-such-run") is None


def test_retrieve_context_no_run_id_returns_empty():
    assert retrieve_context("", "query", 3) == []


def test_retrieve_context_unknown_run_id_returns_empty():
    assert retrieve_context("missing-run", "query", 3) == []


def test_retrieve_context_delegates_to_store():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-5")
    store.add_chunks([Chunk(text="Hello world", source_file="a.md", chunk_index=0)])
    register_store("run-abc", store)
    try:
        assert retrieve_context("run-abc", "Hello world", 1) == ["Hello world"]
    finally:
        clear_store("run-abc")
