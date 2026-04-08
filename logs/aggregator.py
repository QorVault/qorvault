#!/usr/bin/env python3
"""Log aggregator: reads filesystem_events.jsonl and audit_parsed.jsonl,
inserts new records into ai_activity_log in PostgreSQL.

Tracks last-processed position per file in aggregator_cursor.json to
avoid re-inserting duplicates. Designed to run every 60 seconds via
systemd timer or as a oneshot service with a sleep loop.

Usage:
    python3 aggregator.py              # One-shot run
    python3 aggregator.py --loop       # Run every 60 seconds
    python3 aggregator.py --loop --interval 30  # Custom interval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

LOG_DIR = Path(__file__).resolve().parent
CURSOR_FILE = LOG_DIR / "aggregator_cursor.json"
AGGREGATOR_LOG = LOG_DIR / "aggregator.log"

SOURCES = {
    "filesystem_events": LOG_DIR / "filesystem_events.jsonl",
    "audit_parsed": LOG_DIR / "audit_parsed.jsonl",
}

# SECURITY: No credentials are hardcoded here. This service
# requires environment variables or a .env file.
# See .env.example for required variables.
load_dotenv(LOG_DIR.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(AGGREGATOR_LOG),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)


def load_env_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Priority:
      1. Individual POSTGRES_* environment variables
      2. DATABASE_URL environment variable
      3. RuntimeError listing which variables are missing
    """
    host = os.environ.get("POSTGRES_HOST")
    port = os.environ.get("POSTGRES_PORT")
    db = os.environ.get("POSTGRES_DB")
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")

    if all([host, port, db, user, password]):
        return f"host={host} port={port} dbname={db} user={user} password={password}"

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url

    required = {
        "POSTGRES_HOST": host,
        "POSTGRES_PORT": port,
        "POSTGRES_DB": db,
        "POSTGRES_USER": user,
        "POSTGRES_PASSWORD": password,
    }
    missing = [name for name, val in required.items() if not val]
    raise RuntimeError(
        f"Database credentials not found. Set these environment variables "
        f"(or add them to .env): {', '.join(missing)}. "
        f"Alternatively, set DATABASE_URL. See .env.example for required variables."
    )


def load_cursors() -> dict[str, int]:
    """Load byte offsets for each source file."""
    if CURSOR_FILE.exists():
        try:
            return json.loads(CURSOR_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_cursors(cursors: dict[str, int]) -> None:
    """Save byte offsets."""
    try:
        CURSOR_FILE.write_text(json.dumps(cursors, indent=2))
    except OSError as exc:
        logger.warning("Failed to save cursors: %s", exc)


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Read new lines from a JSONL file starting at byte offset.
    Returns (lines, new_offset).
    """
    if not path.exists():
        return [], offset

    file_size = path.stat().st_size
    if file_size <= offset:
        return [], offset

    lines = []
    with open(path) as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
        new_offset = f.tell()

    return lines, new_offset


def parse_fs_event(raw: str) -> dict | None:
    """Parse a filesystem event JSON line into an ai_activity_log record."""
    try:
        evt = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return {
        "session_id": str(uuid.uuid4()),  # No session context for fs events
        "session_start": evt.get("timestamp", datetime.now(UTC).isoformat()),
        "action_type": f"fs_{evt.get('event_type', 'unknown')}",
        "action_timestamp": evt.get("timestamp", datetime.now(UTC).isoformat()),
        "file_path": evt.get("file_path"),
        "source": "inotifywait",
        "raw_event": raw,
        "initiated_by": "filesystem",
    }


def parse_audit_event(raw: str) -> dict | None:
    """Parse an audit event JSON line into an ai_activity_log record."""
    try:
        evt = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return {
        "session_id": str(uuid.uuid4()),
        "session_start": evt.get("timestamp", datetime.now(UTC).isoformat()),
        "action_type": f"audit_{evt.get('event_type', 'unknown')}",
        "action_timestamp": evt.get("timestamp", datetime.now(UTC).isoformat()),
        "file_path": evt.get("file_path"),
        "command_executed": evt.get("command"),
        "success": evt.get("success"),
        "source": "auditd",
        "raw_event": raw,
        "initiated_by": evt.get("audit_key", "auditd"),
    }


PARSERS = {
    "filesystem_events": parse_fs_event,
    "audit_parsed": parse_audit_event,
}


def insert_records(records: list[dict], dsn: str) -> int:
    """Insert records into ai_activity_log. Returns count inserted."""
    if not records:
        return 0

    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed — cannot insert to DB")
        return 0

    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()

        inserted = 0
        for rec in records:
            try:
                cur.execute(
                    """
                    INSERT INTO ai_activity_log (
                        session_id, session_start, action_type,
                        action_timestamp, file_path, command_executed,
                        success, source, raw_event, initiated_by
                    ) VALUES (
                        %(session_id)s::uuid, %(session_start)s, %(action_type)s,
                        %(action_timestamp)s, %(file_path)s, %(command_executed)s,
                        %(success)s, %(source)s, %(raw_event)s::jsonb,
                        %(initiated_by)s
                    )
                """,
                    {
                        "session_id": rec["session_id"],
                        "session_start": rec["session_start"],
                        "action_type": rec["action_type"],
                        "action_timestamp": rec["action_timestamp"],
                        "file_path": rec.get("file_path"),
                        "command_executed": rec.get("command_executed"),
                        "success": rec.get("success"),
                        "source": rec.get("source"),
                        "raw_event": rec.get("raw_event"),
                        "initiated_by": rec.get("initiated_by", "unknown"),
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.warning("Failed to insert record: %s", exc)
                conn.rollback()
                continue

        conn.commit()
        cur.close()
        conn.close()
        return inserted

    except Exception as exc:
        logger.warning("Database connection failed (non-fatal): %s", exc)
        return 0


def run_once(dsn: str) -> None:
    """Single aggregation pass."""
    cursors = load_cursors()
    total_inserted = 0

    for source_name, source_path in SOURCES.items():
        offset = cursors.get(source_name, 0)
        lines, new_offset = read_new_lines(source_path, offset)

        if not lines:
            continue

        parser = PARSERS.get(source_name)
        if not parser:
            continue

        records = []
        for line in lines:
            rec = parser(line)
            if rec:
                records.append(rec)

        if records:
            inserted = insert_records(records, dsn)
            total_inserted += inserted
            logger.info(
                "Source %s: %d lines read, %d records inserted",
                source_name,
                len(lines),
                inserted,
            )

        cursors[source_name] = new_offset

    save_cursors(cursors)

    if total_inserted:
        logger.info("Total: %d records inserted", total_inserted)


def main():
    parser = argparse.ArgumentParser(description="Aggregate logs into PostgreSQL")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between runs in loop mode (default: 60)",
    )
    args = parser.parse_args()

    dsn = load_env_dsn()

    if args.loop:
        logger.info("Starting aggregator loop (interval=%ds)", args.interval)
        while True:
            try:
                run_once(dsn)
            except Exception as exc:
                logger.error("Aggregator error (continuing): %s", exc)
            time.sleep(args.interval)
    else:
        run_once(dsn)


if __name__ == "__main__":
    main()
