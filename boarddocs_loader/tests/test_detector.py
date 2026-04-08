"""Tests for format detection."""

import json

from boarddocs_loader.detector import detect_format, is_attachment_file


def test_detect_structured(tmp_path):
    """Directory with subdirectories → structured."""
    meeting_dir = tmp_path / "meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text("{}")
    item_dir = meeting_dir / "1-01-abc123-item-one"
    item_dir.mkdir()
    (item_dir / "item.json").write_text("{}")

    assert detect_format(meeting_dir) == "structured"


def test_detect_flat(tmp_path):
    """Directory with agenda.txt, no subdirectories → flat."""
    meeting_dir = tmp_path / "meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text("{}")
    (meeting_dir / "agenda.txt").write_text("content")

    assert detect_format(meeting_dir) == "flat"


def test_detect_empty(tmp_path):
    """Directory with only meeting.json → empty."""
    meeting_dir = tmp_path / "meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text("{}")

    assert detect_format(meeting_dir) == "empty"


def test_detect_empty_executive_session(tmp_path):
    """Executive session with only meeting.json is empty, not an error."""
    meeting_dir = tmp_path / "2005-06-21-executive-session"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(
        json.dumps(
            {
                "meeting_id": "86D4LZ63971C",
                "date": "20050621",
                "name": "Executive Session",
                "slug": "2005-06-21-executive-session",
                "files_found": 0,
            }
        )
    )

    assert detect_format(meeting_dir) == "empty"


def test_detect_flat_with_attachments_only(tmp_path):
    """Directory with meeting.json + PDF but no agenda.txt → flat."""
    meeting_dir = tmp_path / "meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text("{}")
    (meeting_dir / "report.pdf").write_bytes(b"%PDF-1.4")

    assert detect_format(meeting_dir) == "flat"


def test_is_attachment_file():
    assert is_attachment_file("report.pdf") is True
    assert is_attachment_file("slides.pptx") is True
    assert is_attachment_file("data.xlsx") is True
    assert is_attachment_file("meeting.json") is False
    assert is_attachment_file("agenda.html") is False
    assert is_attachment_file("agenda.txt") is False
    assert is_attachment_file(".hidden") is False
    assert is_attachment_file("show.ppsx") is True
    assert is_attachment_file("REPORT.PDF") is True
