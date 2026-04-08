"""Tests for JSON and agenda.txt parsing."""

import json
from datetime import date

from boarddocs_loader.parsers import (
    extract_committee_name,
    parse_agenda_txt,
    parse_item_json,
    parse_meeting_json,
)
from tests.conftest import (
    AGENDA_2005_SAMPLE,
    AGENDA_2026_SAMPLE,
    FLAT_MEETING_JSON,
    ITEM_JSON_SAMPLE,
    STRUCTURED_MEETING_JSON,
)


def test_parse_flat_meeting_json(tmp_path):
    """Flat meeting.json parses correctly."""
    p = tmp_path / "meeting.json"
    p.write_text(FLAT_MEETING_JSON)

    meeting = parse_meeting_json(p, "flat")

    assert meeting.date == date(2026, 2, 11)
    assert "Regular Meeting" in meeting.name
    assert meeting.format == "flat"
    assert meeting.meeting_id == "DQU45Y09F4B0"
    assert meeting.committee_id == "A94NQ8610101"
    assert meeting.slug == "2026-02-11-regular-meeting-630-pm"


def test_parse_structured_meeting_json(tmp_path):
    """Structured meeting.json parses correctly."""
    p = tmp_path / "meeting.json"
    p.write_text(STRUCTURED_MEETING_JSON)

    meeting = parse_meeting_json(p, "structured")

    assert meeting.date == date(2026, 1, 28)
    assert meeting.format == "structured"
    assert len(meeting.categories) == 2
    assert meeting.categories[0].category_name == "Opening"
    assert len(meeting.categories[0].items) == 3


def test_parse_meeting_id_from_url(tmp_path):
    """Structured meeting.json extracts meeting_id from URL."""
    p = tmp_path / "meeting.json"
    p.write_text(STRUCTURED_MEETING_JSON)

    meeting = parse_meeting_json(p, "structured")

    assert meeting.meeting_id == "DQHKYP542B0C"


def test_parse_item_json(tmp_path):
    """item.json parses all fields correctly."""
    p = tmp_path / "item.json"
    p.write_text(ITEM_JSON_SAMPLE)

    item = parse_item_json(p)

    assert item.item_id == "DQU2Q30331E4"
    assert item.item_order == "2.01"
    assert item.item_name == "Budget Work Session"
    assert len(item.links) == 1
    assert item.links[0]["filename"] == "Budget Update 2.4.26.pdf"
    assert "Type" in item.inner_html


def test_parse_agenda_txt_2026():
    """2026-era agenda.txt extracts items correctly."""
    items = parse_agenda_txt(AGENDA_2026_SAMPLE)

    assert len(items) >= 3
    assert "Protocol Guidelines" in items[0].subject
    assert "Opening" in items[0].category


def test_parse_agenda_txt_2005():
    """2005-era agenda.txt extracts items correctly."""
    items = parse_agenda_txt(AGENDA_2005_SAMPLE)

    assert len(items) >= 3
    assert "Call To Order" in items[0].subject
    assert "Opening" in items[0].category


def test_parse_empty_type_defaults():
    """Empty type field defaults to 'Information'."""
    items = parse_agenda_txt(AGENDA_2005_SAMPLE)

    # Second item (1.02 Pledge of Allegiance) has empty Type
    pledge_item = next(i for i in items if "Pledge" in i.subject)
    assert pledge_item.type == "Information"


def test_parse_strips_dash_artifacts():
    """\\--- artifacts are removed from parsed values."""
    text = """Subject
    \\--- Test Item \\---

Meeting
    Jan 1, 2026 - Meeting

Category
    \\--- Category \\---

Type
    Information

File Attachments
"""
    items = parse_agenda_txt(text)
    assert len(items) == 1
    assert "---" not in items[0].subject
    assert "---" not in items[0].category


def test_parse_item_with_attachments():
    """File Attachments with content → has_attachments=True."""
    items = parse_agenda_txt(AGENDA_2026_SAMPLE)
    budget_item = next(i for i in items if "Budget" in i.subject)
    assert budget_item.has_attachments is True


def test_parse_item_without_attachments():
    """File Attachments with no content → has_attachments=False."""
    items = parse_agenda_txt(AGENDA_2026_SAMPLE)
    protocol_item = next(i for i in items if "Protocol" in i.subject)
    assert protocol_item.has_attachments is False


def test_date_parsing_yyyymmdd(tmp_path):
    """YYYYMMDD date string parses correctly."""
    p = tmp_path / "meeting.json"
    p.write_text(
        json.dumps(
            {
                "meeting_id": "TEST",
                "date": "20050323",
                "name": "Test",
                "slug": "test",
            }
        )
    )
    meeting = parse_meeting_json(p, "flat")
    assert meeting.date == date(2005, 3, 23)


def test_committee_name_extraction():
    """Committee name extraction from various meeting names."""
    assert extract_committee_name("Regular Meeting - 6:30 p.m.") == "Regular Meeting"
    assert extract_committee_name("Special Meeting 6:00 p.m.") == "Special Meeting"
    assert extract_committee_name("Executive Session 6:30-8:00 pm") == "Executive Session"
    assert extract_committee_name("Work Session - Budget") == "Work Session"
    assert extract_committee_name("Study Session") == "Study Session"
    assert extract_committee_name("Board Retreat") == "Retreat"
    assert extract_committee_name("Workshop on Policy") == "Work Session"
    long_name = "A" * 300
    assert len(extract_committee_name(long_name)) == 255
