# src/parsers/pdf_parser.py
from __future__ import annotations

import pypdf

from src.parsers.base import SourceDocument


def parse_pdf_file(file_path: str) -> SourceDocument:
    try:
        reader = pypdf.PdfReader(file_path)
    except Exception as exc:  # noqa: BLE001 - any pypdf failure must not crash ingestion
        return SourceDocument(
            file_path=file_path, text="", warnings=[f"Could not open PDF {file_path}: {exc}"]
        )

    page_texts: list[str] = []
    warnings: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            page_texts.append(f"[Page {index + 1}]\n{page.extract_text() or ''}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to extract page {index + 1} of {file_path}: {exc}")
    return SourceDocument(file_path=file_path, text="\n\n".join(page_texts), warnings=warnings)
