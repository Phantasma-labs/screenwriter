# src/parsers/chunker.py
from __future__ import annotations

from dataclasses import dataclass

from src.parsers.base import ParsedContext


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int


def chunk_text(text: str, source_file: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(Chunk(text=text[start:end], source_file=source_file, chunk_index=index))
        if end == length:
            break
        start = end - overlap
        index += 1
    return chunks


def chunk_parsed_context(context: ParsedContext, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in context.documents:
        chunks.extend(chunk_text(doc.text, doc.file_path, chunk_size, overlap))
    return chunks
