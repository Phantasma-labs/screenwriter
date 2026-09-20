# tests/webapp/test_archive.py
from __future__ import annotations

import zipfile
from io import BytesIO

from src.webapp.archive import build_markdown_zip


def test_build_markdown_zip_contains_every_file_with_correct_content():
    files = {"Overview.md": "OVERVIEW", "Script.md": "SCRIPT"}
    data = build_markdown_zip(files)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert set(zf.namelist()) == {"Overview.md", "Script.md"}
        assert zf.read("Overview.md").decode("utf-8") == "OVERVIEW"
        assert zf.read("Script.md").decode("utf-8") == "SCRIPT"
