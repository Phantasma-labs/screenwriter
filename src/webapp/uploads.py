# src/webapp/uploads.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def save_uploaded_files(uploaded_files: list[UploadedFileLike], dest_dir: Path) -> list[str]:
    saved_paths: list[str] = []
    for uploaded_file in uploaded_files:
        target = dest_dir / uploaded_file.name
        target.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(target))
    return saved_paths
