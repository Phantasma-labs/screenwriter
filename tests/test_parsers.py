# tests/test_parsers.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.parsers.base import ParserError, parse_context_files
from src.parsers.pdf_parser import parse_pdf_file
from src.parsers.text_parser import parse_text_file


def test_parse_text_file_reads_markdown(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Heading\n\nSome body text.", encoding="utf-8")
    doc = parse_text_file(str(md_file))
    assert doc.text == "# Heading\n\nSome body text."
    assert doc.warnings == []


def test_parse_text_file_missing_file_returns_warning():
    doc = parse_text_file("does/not/exist.md")
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert "Could not read" in doc.warnings[0]


def test_parse_context_files_combines_multiple_sources(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("Content A", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("Content B", encoding="utf-8")
    context = parse_context_files([str(a), str(b)])
    assert "Content A" in context.combined_text
    assert "Content B" in context.combined_text
    assert len(context.documents) == 2
    assert context.warnings == []


def test_parse_context_files_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_text("not text")
    with pytest.raises(ParserError):
        parse_context_files([str(bad)])


def _make_fake_page(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def test_parse_pdf_file_extracts_pages():
    fake_reader = MagicMock()
    fake_reader.pages = [_make_fake_page("Page one text"), _make_fake_page("Page two text")]
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", return_value=fake_reader):
        doc = parse_pdf_file("fake.pdf")
    assert "[Page 1]" in doc.text
    assert "Page one text" in doc.text
    assert "[Page 2]" in doc.text
    assert "Page two text" in doc.text
    assert doc.warnings == []


def test_parse_pdf_file_handles_open_failure():
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", side_effect=OSError("corrupt file")):
        doc = parse_pdf_file("broken.pdf")
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert "Could not open PDF" in doc.warnings[0]


def test_parse_pdf_file_handles_page_extraction_failure():
    bad_page = MagicMock()
    bad_page.extract_text.side_effect = RuntimeError("bad page")
    fake_reader = MagicMock()
    fake_reader.pages = [bad_page]
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", return_value=fake_reader):
        doc = parse_pdf_file("fake.pdf")
    assert doc.text == ""
    assert "Failed to extract page 1" in doc.warnings[0]
