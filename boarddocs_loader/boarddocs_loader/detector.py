"""Detect meeting directory format: structured, flat, or empty."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

ATTACHMENT_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".ppt", ".pptx", ".ppsx", ".xls", ".xlsx", ".pps"})
IGNORED_EXTENSIONS = frozenset({".json", ".html", ".htm", ".txt"})


def detect_format(meeting_dir: Path) -> Literal["structured", "flat", "empty"]:
    """Detect whether a meeting directory is structured, flat, or empty.

    - structured: has subdirectories (item directories)
    - flat: has agenda.txt or top-level attachment files, no subdirectories
    - empty: only meeting.json, nothing else
    """
    has_subdirs = False
    has_agenda_txt = False
    has_other_files = False

    for entry in os.scandir(meeting_dir):
        if entry.is_dir(follow_symlinks=False):
            has_subdirs = True
            break
        name = entry.name
        if name.startswith("."):
            continue
        if name == "meeting.json":
            continue
        if name == "agenda.txt" or name == "agenda.html":
            has_agenda_txt = True
        else:
            ext = Path(name).suffix.lower()
            if ext not in IGNORED_EXTENSIONS:
                has_other_files = True

    if has_subdirs:
        return "structured"
    if has_agenda_txt or has_other_files:
        return "flat"
    return "empty"


def is_attachment_file(filename: str) -> bool:
    """Return True if the file should be treated as an attachment."""
    if filename.startswith("."):
        return False
    ext = Path(filename).suffix.lower()
    return ext in ATTACHMENT_EXTENSIONS
