# src/parsers/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class ParserError(RuntimeError):
    """Raised for unsupported file extensions."""


@dataclass
class SourceDocument:
    file_path: str
    text: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedContext:
    documents: list[SourceDocument]
    combined_text: str
    warnings: list[str]


def parse_context_files(file_paths: list[str]) -> ParsedContext:
    from src.parsers.pdf_parser import parse_pdf_file
    from src.parsers.text_parser import parse_text_file

    documents: list[SourceDocument] = []
    warnings: list[str] = []
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ParserError(
                f"Unsupported file type '{ext}' for {file_path}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        doc = parse_pdf_file(file_path) if ext == ".pdf" else parse_text_file(file_path)
        documents.append(doc)
        warnings.extend(doc.warnings)

    combined_text = "\n\n".join(
        f"### Source: {doc.file_path}\n{doc.text}" for doc in documents if doc.text
    )
    return ParsedContext(documents=documents, combined_text=combined_text, warnings=warnings)
