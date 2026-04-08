"""Tests for DataLoader directory walking and document creation."""

import json
from unittest.mock import MagicMock

from boarddocs_loader.loader import DataLoader
from tests.conftest import (
    FLAT_MEETING_JSON,
    ITEM_JSON_SAMPLE,
    STRUCTURED_MEETING_JSON,
)


def _make_structured_dir(tmp_path, meeting_json=STRUCTURED_MEETING_JSON, item_count=3, attach_to_first=False):
    """Create a structured meeting dir with N items."""
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    meeting_dir = root / "2026-01-28-meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(meeting_json)

    items_data = json.loads(ITEM_JSON_SAMPLE)
    for i in range(item_count):
        item_id = f"ITEM{i:04d}"
        items_data_copy = dict(items_data, itemId=item_id, itemOrder=f"1.{i+1:02d}", itemName=f"Item {i+1}")
        subdir = meeting_dir / f"1-{i+1:02d}-{item_id.lower()}-item-{i+1}"
        subdir.mkdir()
        (subdir / "item.json").write_text(json.dumps(items_data_copy))
        if attach_to_first and i == 0:
            (subdir / "report.pdf").write_bytes(b"%PDF-1.4 test")
    return root


def _make_flat_dir(tmp_path, with_agenda=True, with_pdf=False):
    """Create a flat meeting dir."""
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    meeting_dir = root / "2026-02-11-meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(FLAT_MEETING_JSON)
    if with_agenda:
        (meeting_dir / "agenda.txt").write_text(
            "Subject\n    Test Item\n\nMeeting\n    Test\n\nCategory\n    1. Test\n\nType\n    Info\n\nFile Attachments\n"
        )
        (meeting_dir / "agenda.html").write_text("<html>test</html>")
    if with_pdf:
        (meeting_dir / "report.pdf").write_bytes(b"%PDF-1.4 test content")
    return root


def _make_empty_dir(tmp_path):
    """Create an empty meeting dir."""
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    meeting_dir = root / "2005-06-21-exec-session"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(
        json.dumps(
            {
                "meeting_id": "86D4LZ63971C",
                "date": "20050621",
                "name": "Executive Session",
                "slug": "2005-06-21-executive-session",
                "files_found": 0,
                "source_url": "https://example.com",
            }
        )
    )
    return root


# ── Tests ────────────────────────────────────────────────────────────


def test_structured_meeting_creates_item_records(tmp_path):
    """Structured meeting creates one agenda_item per item subdir."""
    root = _make_structured_dir(tmp_path, item_count=3)
    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.meetings_processed == 1
    assert stats.agenda_items_created == 3


def test_structured_meeting_creates_attachment_records(tmp_path):
    """PDF in item subdir creates an attachment record."""
    root = _make_structured_dir(tmp_path, item_count=2, attach_to_first=True)
    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.attachment_records_created == 1


def test_flat_meeting_creates_agenda_record(tmp_path):
    """Flat meeting creates 1 agenda + attachment records."""
    root = _make_flat_dir(tmp_path, with_agenda=True, with_pdf=True)
    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.agenda_records_created == 1
    assert stats.attachment_records_created == 1


def test_empty_meeting_creates_agenda_record_not_error(tmp_path, caplog):
    """Empty meeting creates 1 agenda record with status 'complete', no errors."""
    root = _make_empty_dir(tmp_path)
    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.agenda_records_created == 1
    assert stats.errors == 0
    assert "ERROR" not in caplog.text


def test_external_id_structured(tmp_path):
    """Structured agenda_item external_id format: {meeting_id}_{item_id}."""
    root = _make_structured_dir(tmp_path, item_count=1)
    records: list = []

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    agenda_items = [r for r in records if r.document_type == "agenda_item"]
    assert len(agenda_items) == 1
    eid = agenda_items[0].external_id
    assert "_" in eid
    parts = eid.split("_", 1)
    assert len(parts) == 2
    assert parts[0] == "DQHKYP542B0C"  # meeting_id from structured JSON


def test_external_id_attachment_structured(tmp_path):
    """Structured attachment external_id: {meeting_id}_{item_id}_{filename}."""
    root = _make_structured_dir(tmp_path, item_count=1, attach_to_first=True)
    records: list = []

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    attachments = [r for r in records if r.document_type == "attachment"]
    assert len(attachments) == 1
    eid = attachments[0].external_id
    assert eid.endswith("_report.pdf")
    assert eid.startswith("DQHKYP542B0C_")


def test_external_id_flat_agenda(tmp_path):
    """Flat agenda external_id: {meeting_id}_agenda."""
    root = _make_flat_dir(tmp_path)
    records: list = []

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    agendas = [r for r in records if r.document_type == "agenda"]
    assert len(agendas) == 1
    assert agendas[0].external_id == "DQU45Y09F4B0_agenda"


def test_category_lookup(tmp_path):
    """Category name populated from meeting.json categories array."""
    root = _make_structured_dir(tmp_path, item_count=1)
    records: list = []

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    # Items won't match the category lookup since we use synthetic item_ids
    # but the field should exist
    items = [r for r in records if r.document_type == "agenda_item"]
    assert len(items) == 1
    assert "category_name" in items[0].metadata


def test_inner_html_text_extraction(tmp_path):
    """InnerHtml is parsed for plain text, goals, and item type."""
    records: list = []
    root = _make_structured_dir(tmp_path, item_count=1)

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    items = [r for r in records if r.document_type == "agenda_item"]
    assert len(items) == 1
    meta = items[0].metadata
    assert meta["item_type"] == "Discussion"
    assert "Goal 1: Test Goal" in meta["item_goals"]
    assert "Goal 2: Another Goal" in meta["item_goals"]
    assert items[0].content_text  # should have extracted text


def test_idempotent_skip(tmp_path):
    """DB returning rowcount=0 (conflict) is counted as skip, not error."""
    root = _make_flat_dir(tmp_path)
    mock_db = MagicMock()
    mock_db.insert_document.return_value = False  # ON CONFLICT DO NOTHING

    loader = DataLoader(data_dir=root, tenant="kent_sd", db=mock_db)
    stats = loader.run()

    assert stats.skipped >= 1
    assert stats.errors == 0


def test_corrupt_item_json_continues(tmp_path, caplog):
    """Invalid JSON in one item.json logs warning and continues."""
    root = tmp_path / "data"
    root.mkdir()
    meeting_dir = root / "2026-01-28-meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(STRUCTURED_MEETING_JSON)

    # Good item
    good_dir = meeting_dir / "1-01-good-item"
    good_dir.mkdir()
    (good_dir / "item.json").write_text(ITEM_JSON_SAMPLE)

    # Bad item
    bad_dir = meeting_dir / "2-01-bad-item"
    bad_dir.mkdir()
    (bad_dir / "item.json").write_text("{invalid json!!!")

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.agenda_items_created == 1  # good item processed
    assert stats.errors == 1  # bad item counted as error
    assert "WARNING" in caplog.text or "Invalid item.json" in caplog.text


def test_limit_flag(tmp_path):
    """Limit flag restricts number of meetings processed."""
    root = tmp_path / "data"
    root.mkdir()
    for i in range(20):
        d = root / f"2024-01-{i+1:02d}-meeting"
        d.mkdir()
        (d / "meeting.json").write_text(
            json.dumps(
                {
                    "meeting_id": f"M{i:04d}",
                    "date": f"2024-01-{i+1:02d}",
                    "name": f"Meeting {i+1}",
                    "slug": f"meeting-{i+1}",
                }
            )
        )

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True, limit=5)
    stats = loader.run()

    assert stats.meetings_processed == 5


def test_dry_run_no_db_calls(tmp_path):
    """Dry run never calls db.insert_document."""
    root = _make_flat_dir(tmp_path, with_pdf=True)
    mock_db = MagicMock()

    loader = DataLoader(data_dir=root, tenant="kent_sd", db=mock_db, dry_run=True)
    loader.run()

    mock_db.insert_document.assert_not_called()


def test_format_report(tmp_path):
    """Format report counts structured, flat, and empty."""
    root = tmp_path / "data"
    root.mkdir()

    # Structured
    sd = root / "structured-meeting"
    sd.mkdir()
    (sd / "meeting.json").write_text("{}")
    item = sd / "1-01-item"
    item.mkdir()
    (item / "item.json").write_text("{}")

    # Flat
    fd = root / "flat-meeting"
    fd.mkdir()
    (fd / "meeting.json").write_text("{}")
    (fd / "agenda.txt").write_text("content")

    # Empty
    ed = root / "empty-meeting"
    ed.mkdir()
    (ed / "meeting.json").write_text("{}")

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.format_report()

    assert stats.structured_count == 1
    assert stats.flat_count == 1
    assert stats.empty_count == 1
    assert stats.meetings_processed == 3


def test_missing_meeting_json_skipped(tmp_path, caplog):
    """Directory without meeting.json is skipped."""
    root = tmp_path / "data"
    root.mkdir()
    d = root / "no-json-meeting"
    d.mkdir()
    (d / "agenda.txt").write_text("content")

    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    stats = loader.run()

    assert stats.meetings_processed == 0


def test_attachment_extension_filter(tmp_path):
    """Only supported extensions become attachment records."""
    root = tmp_path / "data"
    root.mkdir()
    meeting_dir = root / "2026-test-meeting"
    meeting_dir.mkdir()
    (meeting_dir / "meeting.json").write_text(FLAT_MEETING_JSON)
    (meeting_dir / "agenda.txt").write_text(
        "Subject\n    Test\n\nMeeting\n    T\n\nCategory\n    C\n\nType\n    I\n\nFile Attachments\n"
    )
    # Supported
    (meeting_dir / "report.pdf").write_bytes(b"%PDF")
    (meeting_dir / "doc.doc").write_bytes(b"doc")
    # Not supported
    (meeting_dir / "meeting.json").exists()  # already exists
    (meeting_dir / "agenda.html").write_text("<html></html>")
    (meeting_dir / "notes.txt").write_text("notes")

    records: list = []
    loader = DataLoader(data_dir=root, tenant="kent_sd", dry_run=True)
    orig_insert = loader._insert

    def capture(doc, kind):
        records.append(doc)
        orig_insert(doc, kind)

    loader._insert = capture
    loader.run()

    attachments = [r for r in records if r.document_type == "attachment"]
    assert len(attachments) == 2
    names = {a.title for a in attachments}
    assert names == {"report.pdf", "doc.doc"}
