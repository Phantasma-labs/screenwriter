# tests/webapp/test_uploads.py
from __future__ import annotations

from src.webapp.uploads import save_uploaded_files


class _FakeUploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_save_uploaded_files_writes_bytes_under_original_name(tmp_path):
    uploads = [_FakeUploadedFile("notes.md", b"# Notes"), _FakeUploadedFile("brief.txt", b"Brief")]
    saved = save_uploaded_files(uploads, tmp_path)
    assert saved == [str(tmp_path / "notes.md"), str(tmp_path / "brief.txt")]
    assert (tmp_path / "notes.md").read_bytes() == b"# Notes"
    assert (tmp_path / "brief.txt").read_bytes() == b"Brief"


def test_save_uploaded_files_preserves_order(tmp_path):
    uploads = [_FakeUploadedFile(f"{i}.txt", str(i).encode()) for i in range(3)]
    saved = save_uploaded_files(uploads, tmp_path)
    assert saved == [str(tmp_path / "0.txt"), str(tmp_path / "1.txt"), str(tmp_path / "2.txt")]


def test_save_uploaded_files_empty_list_returns_empty_list(tmp_path):
    assert save_uploaded_files([], tmp_path) == []
