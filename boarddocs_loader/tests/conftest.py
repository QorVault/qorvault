"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Sample JSON strings ──────────────────────────────────────────────

FLAT_MEETING_JSON = json.dumps(
    {
        "meeting_id": "DQU45Y09F4B0",
        "date": "20260211",
        "name": "Regular Meeting - 6:30 p.m.",
        "unid": "7DBB82BEA53DA04985258D900009F4B0",
        "committee_id": "A94NQ8610101",
        "slug": "2026-02-11-regular-meeting-630-pm",
        "files_found": 21,
        "scraped_at": "2026-02-22T07:02:41.345627",
        "source_url": "https://go.boarddocs.com/wa/ksdwa/Board.nsf/Public",
    }
)

STRUCTURED_MEETING_JSON = json.dumps(
    {
        "date": "2026-01-28",
        "meetingSlug": "2026-01-28-regular-meeting-6-30-p-m-",
        "meetingType": "Regular Meeting - 6:30 p.m.",
        "meetingUrl": "https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=DQHKYP542B0C",
        "categories": [
            {
                "categoryId": "DQHKYQ542B0D",
                "categoryOrder": "1",
                "categoryName": "Opening",
                "items": [
                    {"itemId": "DQHKYR542B16", "itemOrder": "1.01", "itemName": "Protocol Guidelines"},
                    {"itemId": "DQHKYS542B19", "itemOrder": "1.02", "itemName": "Call to Order"},
                    {"itemId": "DQHKYV542B2E", "itemOrder": "1.03", "itemName": "Roll Call"},
                ],
            },
            {
                "categoryId": "DQHKYW542B31",
                "categoryOrder": "2",
                "categoryName": "Student Presentations",
                "items": [
                    {
                        "itemId": "DQHUAA7B15EB",
                        "itemOrder": "2.01",
                        "itemName": "Student Presentation - Meridian Elementary",
                    },
                ],
            },
        ],
        "scrapedAt": "2026-02-04T09:24:23.013Z",
    }
)

ITEM_JSON_SAMPLE = json.dumps(
    {
        "itemId": "DQU2Q30331E4",
        "itemOrder": "2.01",
        "itemName": "Budget Work Session",
        "itemSlug": "2-01-dqu2q30331e4-budget-work-session",
        "itemUrl": "https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=DQU2Q30331E4",
        "links": [
            {
                "order": "00001",
                "unique": "DQVAHX27D3F1",
                "href": "https://example.com/file.pdf",
                "text": "Budget Update 2.4.26.pdf (1,613 KB)",
                "filename": "Budget Update 2.4.26.pdf",
            }
        ],
        "innerHtml": (
            '<dl class="row"><dt class="col leftcol">Type</dt>'
            '<dd class="col rightcol">Discussion</dd></dl>'
            '<a class="goal" unique="G1" role="link">'
            '<div class="name">Goal 1: Test Goal</div></a>'
            '<a class="goal" unique="G2" role="link">'
            '<div class="name">Goal 2: Another Goal</div></a>'
            "<p><strong>Some bold text</strong> and regular text.</p>"
        ),
    }
)

# ── Agenda text samples ──────────────────────────────────────────────

AGENDA_2026_SAMPLE = """1

Wednesday, February 11, 2026

Regular Meeting - 6:30 p.m.

This Regular Board meeting will be held in person.
\\---
PUBLIC COMMENT SUBMISSION
\\---
Public comments for the Regular Board meeting will be accepted.

1. Opening

Subject
    1.01 Protocol Guidelines

Meeting
    Feb 11, 2026 - Regular Meeting - 6:30 p.m.

Category
    1. Opening

Type
    Information

File Attachments

Subject
    1.02 Call to Order

Meeting
    Feb 11, 2026 - Regular Meeting - 6:30 p.m.

Category
    1. Opening

Type
    Information

File Attachments

Subject
    1.03 Roll Call

Meeting
    Feb 11, 2026 - Regular Meeting - 6:30 p.m.

Category
    1. Opening

Type
    Information

File Attachments

Subject
    2.01 Budget Discussion

Meeting
    Feb 11, 2026 - Regular Meeting - 6:30 p.m.

Category
    2. Reports

Type
    Discussion

File Attachments
Budget_Report_2026.pdf
"""

AGENDA_2005_SAMPLE = """0

Wednesday, March 23, 2005

Regular Meeting 7:00 p.m.

1.0 Opening

Subject
     1.01 Call To Order

Meeting
    Mar 23, 2005 - Regular Meeting 7:00 p.m.

Category
     1.0 Opening

Type
    Information

File Attachments

Subject
     1.02 Pledge of Allegiance

Meeting
    Mar 23, 2005 - Regular Meeting 7:00 p.m.

Category
     1.0 Opening

Type


File Attachments

Subject
     1.03 Welcome

Meeting
    Mar 23, 2005 - Regular Meeting 7:00 p.m.

Category
     1.0 Opening

Type


File Attachments

Subject
     2.01 Transportation Vehicle Fund

Meeting
    Mar 23, 2005 - Regular Meeting 7:00 p.m.

Category
     2.0 Presentations

Type
    Presentations

File Attachments
"""


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_meeting_dir(tmp_path):
    """Create a temporary meeting directory with meeting.json."""

    def _create(
        meeting_json_str: str, files: dict[str, str] | None = None, subdirs: dict[str, dict[str, str]] | None = None
    ) -> Path:
        meeting_dir = tmp_path / "test-meeting"
        meeting_dir.mkdir(exist_ok=True)
        (meeting_dir / "meeting.json").write_text(meeting_json_str)
        if files:
            for name, content in files.items():
                (meeting_dir / name).write_text(content)
        if subdirs:
            for dirname, dir_files in subdirs.items():
                subdir = meeting_dir / dirname
                subdir.mkdir(exist_ok=True)
                for fname, fcontent in dir_files.items():
                    (subdir / fname).write_text(fcontent)
        return meeting_dir

    return _create
