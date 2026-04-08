"""PostgreSQL connection and insert functions."""

from __future__ import annotations

import json
import logging
import re

import psycopg2
import psycopg2.extras

from .models import DocumentRecord

logger = logging.getLogger(__name__)

# Register UUID adapter
psycopg2.extras.register_uuid()

INSERT_SQL = """
INSERT INTO documents (
    tenant_id, external_id, document_type, title,
    content_raw, content_text, source_url, file_path,
    meeting_date, committee_name, meeting_id, agenda_item_id,
    processing_status, metadata
) VALUES (
    %(tenant_id)s, %(external_id)s, %(document_type)s, %(title)s,
    %(content_raw)s, %(content_text)s, %(source_url)s, %(file_path)s,
    %(meeting_date)s, %(committee_name)s, %(meeting_id)s, %(agenda_item_id)s,
    %(processing_status)s, %(metadata)s
) ON CONFLICT (tenant_id, external_id) DO NOTHING
"""


def _sanitize_dsn(dsn: str) -> str:
    """Remove password from DSN for safe logging."""
    return re.sub(r"://[^:]+:[^@]+@", "://***:***@", dsn)


class Database:
    """Thin wrapper around psycopg2 for document insertion."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: psycopg2.extensions.connection | None = None

    def connect(self) -> None:
        """Open a connection to PostgreSQL."""
        logger.info("Connecting to PostgreSQL at %s", _sanitize_dsn(self._dsn))
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False
        logger.info("Connected successfully")

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def insert_document(self, doc: DocumentRecord) -> bool:
        """Insert a document record. Returns True if inserted, False if skipped."""
        assert self._conn is not None, "Database not connected"
        cur = self._conn.cursor()
        try:
            params = {
                "tenant_id": doc.tenant_id,
                "external_id": doc.external_id,
                "document_type": doc.document_type,
                "title": doc.title,
                "content_raw": doc.content_raw,
                "content_text": doc.content_text,
                "source_url": doc.source_url,
                "file_path": doc.file_path,
                "meeting_date": doc.meeting_date,
                "committee_name": doc.committee_name,
                "meeting_id": doc.meeting_id,
                "agenda_item_id": doc.agenda_item_id,
                "processing_status": doc.processing_status,
                "metadata": json.dumps(doc.metadata),
            }
            cur.execute(INSERT_SQL, params)
            self._conn.commit()
            return cur.rowcount > 0
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()
