# tests/test_chunker.py
from __future__ import annotations

import pytest

from src.parsers.base import ParsedContext, SourceDocument
from src.parsers.chunker import chunk_parsed_context, chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", "a.md", chunk_size=10, overlap=2) == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    chunks = chunk_text("short text", "a.md", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].chunk_index == 0


def test_chunk_text_splits_with_overlap():
    text = "a" * 25
    chunks = chunk_text(text, "a.md", chunk_size=10, overlap=3)
    assert [c.text for c in chunks] == ["a" * 10, "a" * 10, "a" * 10, "a" * 4]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert all(c.source_file == "a.md" for c in chunks)


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", "a.md", chunk_size=5, overlap=5)


def test_chunk_parsed_context_chunks_every_document():
    context = ParsedContext(
        documents=[
            SourceDocument(file_path="a.md", text="a" * 15),
            SourceDocument(file_path="b.md", text="b" * 5),
        ],
        combined_text="",
        warnings=[],
    )
    chunks = chunk_parsed_context(context, chunk_size=10, overlap=2)
    sources = {c.source_file for c in chunks}
    assert sources == {"a.md", "b.md"}
