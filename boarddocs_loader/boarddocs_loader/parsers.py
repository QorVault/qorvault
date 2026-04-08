"""Parsers for meeting.json, item.json, and agenda.txt."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .models import (
    AgendaItem,
    CategoryRecord,
    ItemJsonRecord,
    ItemRef,
    MeetingRecord,
)

logger = logging.getLogger(__name__)


# ── Meeting JSON parsing ─────────────────────────────────────────────


def parse_meeting_json(path: Path, fmt: str) -> MeetingRecord:
    """Parse a meeting.json file into a MeetingRecord.

    Detects the JSON schema by checking for the 'categories' key.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    is_structured_schema = "categories" in raw
    meeting_dir = str(path.parent)

    if is_structured_schema:
        return _parse_structured_meeting(raw, fmt, meeting_dir)
    else:
        return _parse_flat_meeting(raw, fmt, meeting_dir)


def _parse_structured_meeting(raw: dict, fmt: str, dir_path: str) -> MeetingRecord:
    """Parse structured-schema meeting.json (has categories array)."""
    meeting_id = _extract_id_from_url(raw.get("meetingUrl", ""))
    meeting_date = _parse_date(raw.get("date", ""))
    scraped_at = _parse_datetime(raw.get("scrapedAt"))

    categories = []
    for cat in raw.get("categories", []):
        items = [
            ItemRef(
                item_id=item["itemId"],
                item_order=item.get("itemOrder", ""),
                item_name=item.get("itemName", ""),
            )
            for item in cat.get("items", [])
        ]
        categories.append(
            CategoryRecord(
                category_id=cat.get("categoryId", ""),
                category_order=cat.get("categoryOrder", ""),
                category_name=cat.get("categoryName", ""),
                items=items,
            )
        )

    return MeetingRecord(
        meeting_id=meeting_id,
        date=meeting_date,
        name=raw.get("meetingType", ""),
        slug=raw.get("meetingSlug", ""),
        source_url=raw.get("meetingUrl", ""),
        scraped_at=scraped_at,
        format=fmt,
        categories=categories,
        dir_path=dir_path,
    )


def _parse_flat_meeting(raw: dict, fmt: str, dir_path: str) -> MeetingRecord:
    """Parse flat-schema meeting.json (has meeting_id field)."""
    meeting_date = _parse_date(raw.get("date", ""))
    scraped_at = _parse_datetime(raw.get("scraped_at"))

    return MeetingRecord(
        meeting_id=raw.get("meeting_id", ""),
        date=meeting_date,
        name=raw.get("name", ""),
        slug=raw.get("slug", ""),
        source_url=raw.get("source_url", ""),
        committee_id=raw.get("committee_id"),
        unid=raw.get("unid"),
        scraped_at=scraped_at,
        format=fmt,
        files_found=raw.get("files_found"),
        dir_path=dir_path,
    )


def _extract_id_from_url(url: str) -> str:
    """Extract the id= parameter value from a BoardDocs meetingUrl."""
    if not url:
        return ""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    id_vals = params.get("id", [])
    return id_vals[0] if id_vals else ""


def _parse_date(date_str: str) -> date | None:
    """Parse date from either 'YYYYMMDD' or 'YYYY-MM-DD' format."""
    if not date_str:
        return None
    try:
        date_str = date_str.strip()
        if len(date_str) == 8 and date_str.isdigit():
            return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        if "-" in date_str:
            return date.fromisoformat(date_str)
    except (ValueError, IndexError):
        logger.warning("Invalid date format: %s", date_str)
    return None


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse an ISO datetime string."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


# ── Item JSON parsing ────────────────────────────────────────────────


def parse_item_json(path: Path) -> ItemJsonRecord:
    """Parse an item.json file into an ItemJsonRecord."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ItemJsonRecord(
        item_id=raw.get("itemId", ""),
        item_order=raw.get("itemOrder", ""),
        item_name=raw.get("itemName", ""),
        item_slug=raw.get("itemSlug", ""),
        item_url=raw.get("itemUrl", ""),
        links=raw.get("links", []),
        inner_html=raw.get("innerHtml", ""),
    )


# ── Agenda text parsing ─────────────────────────────────────────────


def parse_agenda_txt(text: str) -> list[AgendaItem]:
    """Parse agenda.txt into a list of AgendaItem objects.

    Handles both 2005-era and 2026-era format variants.
    Skips all content before the first standalone 'Subject' line.
    """
    if not text:
        return []

    lines = text.split("\n")

    # Find the first line that is exactly "Subject" when stripped
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "Subject":
            start_idx = i
            break

    if start_idx is None:
        return []

    # Parse blocks starting from first Subject
    items: list[AgendaItem] = []
    current_field: str | None = None
    subject = ""
    category = ""
    item_type = ""
    file_attachments_content: list[str] = []
    in_file_attachments = False

    def _flush():
        nonlocal subject, category, item_type, file_attachments_content, in_file_attachments
        if subject:
            has_attach = any(ln.strip() and ln.strip() != "---" for ln in file_attachments_content)
            items.append(
                AgendaItem(
                    subject=_clean(subject),
                    category=_clean(category),
                    type=_clean(item_type) or "Information",
                    has_attachments=has_attach,
                )
            )
        subject = ""
        category = ""
        item_type = ""
        file_attachments_content = []
        in_file_attachments = False

    for i in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if stripped == "Subject":
            # Start of a new block — flush previous
            if subject:
                _flush()
            current_field = "subject"
            in_file_attachments = False
            continue

        if stripped == "Meeting":
            current_field = "meeting"
            in_file_attachments = False
            continue

        if stripped == "Category":
            current_field = "category"
            in_file_attachments = False
            continue

        if stripped == "Type":
            current_field = "type"
            in_file_attachments = False
            continue

        if stripped == "Goals":
            current_field = "goals"
            in_file_attachments = False
            continue

        if stripped == "File Attachments":
            current_field = "file_attachments"
            in_file_attachments = True
            continue

        # Accumulate value for current field (first non-empty indented line)
        if current_field == "subject" and not subject and stripped:
            subject = stripped
            current_field = None
        elif current_field == "category" and not category and stripped:
            category = stripped
            current_field = None
        elif current_field == "type" and not item_type and stripped:
            item_type = stripped
            current_field = None
        elif current_field == "type" and not item_type and stripped == "":
            # Empty type — will default to "Information"
            item_type = ""
            current_field = None
        elif current_field == "meeting":
            # Skip meeting lines
            if stripped:
                current_field = None
        elif current_field == "goals":
            # Skip goals lines
            pass
        elif in_file_attachments:
            file_attachments_content.append(line)

    # Flush the last item
    _flush()

    return items


def _clean(value: str) -> str:
    """Strip whitespace and remove \\--- artifacts."""
    value = value.replace("\\---", "").replace("---", "")
    return value.strip()


# ── Committee name extraction ────────────────────────────────────────

COMMITTEE_PATTERNS = [
    ("Regular Meeting", "Regular Meeting"),
    ("Special Meeting", "Special Meeting"),
    ("Executive Session", "Executive Session"),
    ("Work Session", "Work Session"),
    ("Workshop", "Work Session"),
    ("Study Session", "Study Session"),
    ("Retreat", "Retreat"),
]


def extract_committee_name(meeting_name: str) -> str:
    """Extract a normalized committee/meeting type from the meeting name."""
    for pattern, result in COMMITTEE_PATTERNS:
        if pattern.lower() in meeting_name.lower():
            return result
    return meeting_name[:255]
