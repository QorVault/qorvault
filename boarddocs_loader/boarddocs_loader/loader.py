"""Main DataLoader: orchestrates walking directories and creating records."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db import Database
from .detector import detect_format, is_attachment_file
from .html_utils import extract_goals, extract_item_type, extract_plain_text
from .models import DocumentRecord, MeetingRecord
from .parsers import (
    extract_committee_name,
    parse_agenda_txt,
    parse_item_json,
    parse_meeting_json,
)

logger = logging.getLogger(__name__)


@dataclass
class LoaderStats:
    meetings_processed: int = 0
    structured_count: int = 0
    flat_count: int = 0
    empty_count: int = 0
    agenda_items_created: int = 0
    agenda_records_created: int = 0
    attachment_records_created: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed: float = 0.0


@dataclass
class DataLoader:
    data_dir: Path
    tenant: str
    db: Database | None = None
    dry_run: bool = False
    limit: int = 0
    verbose: bool = False
    stats: LoaderStats = field(default_factory=LoaderStats)

    def run(self) -> LoaderStats:
        """Walk all meeting directories and load documents."""
        start = time.time()
        meeting_dirs = sorted(p for p in self.data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

        if self.limit > 0:
            meeting_dirs = meeting_dirs[: self.limit]

        for meeting_dir in meeting_dirs:
            try:
                self._process_meeting(meeting_dir)
            except Exception as e:
                logger.warning("Error processing %s: %s", meeting_dir.name, e)
                self.stats.errors += 1

        self.stats.elapsed = time.time() - start
        return self.stats

    def format_report(self) -> LoaderStats:
        """Scan all directories and report format counts without loading."""
        meeting_dirs = sorted(p for p in self.data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

        for meeting_dir in meeting_dirs:
            meeting_json = meeting_dir / "meeting.json"
            if not meeting_json.exists():
                continue
            fmt = detect_format(meeting_dir)
            self.stats.meetings_processed += 1
            if fmt == "structured":
                self.stats.structured_count += 1
            elif fmt == "flat":
                self.stats.flat_count += 1
            else:
                self.stats.empty_count += 1

        return self.stats

    def _process_meeting(self, meeting_dir: Path) -> None:
        """Process a single meeting directory."""
        meeting_json = meeting_dir / "meeting.json"
        if not meeting_json.exists():
            logger.warning("No meeting.json in %s, skipping", meeting_dir.name)
            return

        fmt = detect_format(meeting_dir)
        if self.verbose:
            logger.debug("Processing %s (format: %s)", meeting_dir.name, fmt)

        try:
            meeting = parse_meeting_json(meeting_json, fmt)
        except Exception as e:
            logger.warning("Invalid meeting.json in %s: %s", meeting_dir.name, e)
            self.stats.errors += 1
            return

        self.stats.meetings_processed += 1
        if fmt == "structured":
            self.stats.structured_count += 1
            self._process_structured(meeting, meeting_dir)
        elif fmt == "flat":
            self.stats.flat_count += 1
            self._process_flat(meeting, meeting_dir)
        else:
            self.stats.empty_count += 1
            self._process_empty(meeting)

    def _process_structured(self, meeting: MeetingRecord, meeting_dir: Path) -> None:
        """Process a structured meeting with item subdirectories."""
        committee = extract_committee_name(meeting.name)

        # Build category lookup from meeting.json categories
        cat_lookup: dict[str, tuple[str, str]] = {}
        for cat in meeting.categories:
            for item in cat.items:
                cat_lookup[item.item_id] = (cat.category_name, cat.category_order)

        for entry in sorted(meeting_dir.iterdir()):
            if not entry.is_dir():
                continue

            item_json_path = entry / "item.json"
            if not item_json_path.exists():
                if self.verbose:
                    logger.debug("No item.json in %s, skipping subdir", entry.name)
                continue

            try:
                item = parse_item_json(item_json_path)
            except Exception as e:
                logger.warning("Invalid item.json in %s: %s", entry, e)
                self.stats.errors += 1
                continue

            # Find attachments in this item subdirectory
            attachments = [f.name for f in entry.iterdir() if f.is_file() and is_attachment_file(f.name)]

            cat_name, cat_order = cat_lookup.get(item.item_id, (None, None))

            goals = extract_goals(item.inner_html)
            item_type = extract_item_type(item.inner_html)
            content_text = extract_plain_text(item.inner_html)

            doc = DocumentRecord(
                tenant_id=self.tenant,
                external_id=f"{meeting.meeting_id}_{item.item_id}",
                document_type="agenda_item",
                title=item.item_name,
                content_raw=item.inner_html or None,
                content_text=content_text or None,
                source_url=item.item_url,
                file_path=str(entry),
                meeting_date=meeting.date,
                committee_name=committee,
                meeting_id=meeting.meeting_id,
                agenda_item_id=item.item_id,
                processing_status="pending",
                metadata={
                    "meeting_slug": meeting.slug,
                    "item_order": item.item_order,
                    "item_slug": item.item_slug,
                    "category_name": cat_name,
                    "category_order": cat_order,
                    "item_goals": goals,
                    "item_type": item_type,
                    "has_attachments": len(attachments) > 0,
                    "attachment_count": len(attachments),
                    "links": item.links,
                },
            )
            self._insert(doc, "agenda_item")

            # Insert attachment records
            for att_name in attachments:
                att_path = entry / att_name
                try:
                    file_size = att_path.stat().st_size
                except OSError as e:
                    logger.warning("Cannot stat %s: %s", att_path, e)
                    self.stats.errors += 1
                    continue

                att_doc = DocumentRecord(
                    tenant_id=self.tenant,
                    external_id=f"{meeting.meeting_id}_{item.item_id}_{att_name}",
                    document_type="attachment",
                    title=att_name,
                    file_path=str(att_path),
                    meeting_date=meeting.date,
                    committee_name=committee,
                    meeting_id=meeting.meeting_id,
                    agenda_item_id=item.item_id,
                    processing_status="pending",
                    metadata={
                        "meeting_slug": meeting.slug,
                        "file_extension": Path(att_name).suffix.lower(),
                        "file_size_bytes": file_size,
                        "item_order": item.item_order,
                        "item_name": item.item_name,
                    },
                )
                self._insert(att_doc, "attachment")

    def _process_flat(self, meeting: MeetingRecord, meeting_dir: Path) -> None:
        """Process a flat meeting with agenda files and top-level attachments."""
        committee = extract_committee_name(meeting.name)

        # Read agenda content
        agenda_txt_path = meeting_dir / "agenda.txt"
        agenda_html_path = meeting_dir / "agenda.html"
        content_text = None
        content_raw = None
        file_path = None

        if agenda_txt_path.exists():
            try:
                content_text = agenda_txt_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Cannot read %s: %s", agenda_txt_path, e)

        if agenda_html_path.exists():
            try:
                content_raw = agenda_html_path.read_text(encoding="utf-8")
                file_path = str(agenda_html_path)
            except Exception as e:
                logger.warning("Cannot read %s: %s", agenda_html_path, e)

        if file_path is None and agenda_txt_path.exists():
            file_path = str(agenda_txt_path)

        # Parse agenda items from text
        agenda_items = []
        if content_text:
            try:
                parsed = parse_agenda_txt(content_text)
                agenda_items = [item.model_dump() for item in parsed]
            except Exception as e:
                logger.warning("Failed to parse agenda.txt for %s: %s", meeting.meeting_id, e)

        scraped_str = meeting.scraped_at.isoformat() if meeting.scraped_at else None

        doc = DocumentRecord(
            tenant_id=self.tenant,
            external_id=f"{meeting.meeting_id}_agenda",
            document_type="agenda",
            title=meeting.name,
            content_raw=content_raw,
            content_text=content_text,
            source_url=meeting.source_url,
            file_path=file_path,
            meeting_date=meeting.date,
            committee_name=committee,
            meeting_id=meeting.meeting_id,
            processing_status="pending",
            metadata={
                "meeting_slug": meeting.slug,
                "committee_id": meeting.committee_id,
                "unid": meeting.unid,
                "scraped_at": scraped_str,
                "files_found": meeting.files_found,
                "agenda_items": agenda_items,
            },
        )
        self._insert(doc, "agenda")

        # Insert attachment records for top-level files
        for entry in sorted(meeting_dir.iterdir()):
            if not entry.is_file():
                continue
            if not is_attachment_file(entry.name):
                continue

            try:
                file_size = entry.stat().st_size
            except OSError as e:
                logger.warning("Cannot stat %s: %s", entry, e)
                self.stats.errors += 1
                continue

            att_doc = DocumentRecord(
                tenant_id=self.tenant,
                external_id=f"{meeting.meeting_id}_{entry.name}",
                document_type="attachment",
                title=entry.name,
                file_path=str(entry),
                meeting_date=meeting.date,
                committee_name=committee,
                meeting_id=meeting.meeting_id,
                processing_status="pending",
                metadata={
                    "meeting_slug": meeting.slug,
                    "file_extension": entry.suffix.lower(),
                    "file_size_bytes": file_size,
                    "item_order": None,
                    "item_name": None,
                },
            )
            self._insert(att_doc, "attachment")

    def _process_empty(self, meeting: MeetingRecord) -> None:
        """Process an empty meeting (only meeting.json)."""
        committee = extract_committee_name(meeting.name)

        doc = DocumentRecord(
            tenant_id=self.tenant,
            external_id=f"{meeting.meeting_id}_agenda",
            document_type="agenda",
            title=meeting.name,
            content_raw=None,
            content_text=None,
            source_url=meeting.source_url,
            meeting_date=meeting.date,
            committee_name=committee,
            meeting_id=meeting.meeting_id,
            processing_status="complete",
            metadata={"meeting_slug": meeting.slug, "empty": True},
        )
        self._insert(doc, "agenda")

    def _insert(self, doc: DocumentRecord, kind: str) -> None:
        """Insert a document record into the database."""
        if self.dry_run:
            if self.verbose:
                logger.debug("DRY RUN: would insert %s %s", kind, doc.external_id)
            if kind == "agenda_item":
                self.stats.agenda_items_created += 1
            elif kind == "agenda":
                self.stats.agenda_records_created += 1
            elif kind == "attachment":
                self.stats.attachment_records_created += 1
            return

        assert self.db is not None, "Database not connected and not in dry-run mode"
        try:
            inserted = self.db.insert_document(doc)
            if inserted:
                if kind == "agenda_item":
                    self.stats.agenda_items_created += 1
                elif kind == "agenda":
                    self.stats.agenda_records_created += 1
                elif kind == "attachment":
                    self.stats.attachment_records_created += 1
            else:
                self.stats.skipped += 1
        except Exception as e:
            logger.warning("Insert failed for %s: %s", doc.external_id, e)
            self.stats.errors += 1
