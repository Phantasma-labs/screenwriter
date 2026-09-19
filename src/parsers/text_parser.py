# src/parsers/text_parser.py
from __future__ import annotations

from pathlib import Path

from src.parsers.base import SourceDocument


def parse_text_file(file_path: str) -> SourceDocument:
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return SourceDocument(
            file_path=file_path, text="", warnings=[f"Could not read {file_path}: {exc}"]
        )
    return SourceDocument(file_path=file_path, text=text)
