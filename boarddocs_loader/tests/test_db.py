"""Tests for database insert logic (mocked DB)."""

import logging
from unittest.mock import MagicMock, patch

from boarddocs_loader.db import Database, _sanitize_dsn
from boarddocs_loader.models import DocumentRecord


def test_insert_document_idempotent():
    """ON CONFLICT DO NOTHING (rowcount=0) returns False, no exception."""
    db = Database("postgresql://boarddocs:secret@host:5432/boarddocs")

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 0
    mock_conn.cursor.return_value = mock_cursor
    db._conn = mock_conn

    doc = DocumentRecord(
        tenant_id="kent_sd",
        external_id="TEST_agenda",
        document_type="agenda",
        title="Test Meeting",
    )

    result = db.insert_document(doc)
    assert result is False  # skipped, not error
    mock_cursor.execute.assert_called_once()


def test_dsn_never_logged(caplog):
    """DSN string with password must never appear in log output."""
    dsn = "postgresql://boarddocs:supersecretpassword@10.0.0.5:5432/boarddocs"

    with caplog.at_level(logging.DEBUG):
        sanitized = _sanitize_dsn(dsn)
        # Verify sanitization works
        assert "supersecretpassword" not in sanitized
        assert "***" in sanitized

        # Attempt a connection that will fail (mocked to not actually connect)
        db = Database(dsn)
        with patch("boarddocs_loader.db.psycopg2.connect", side_effect=Exception("connection refused")):
            try:
                db.connect()
            except Exception:
                pass

        # Check no log line contains the password
        for record in caplog.records:
            assert "supersecretpassword" not in record.getMessage()
            assert dsn not in record.getMessage()
