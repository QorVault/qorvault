#!/usr/bin/env python3
"""Session logging class for AI activity tracking.

Provides a SessionLogger class that can be imported or used via CLI.
Tracks session start/end, model info, and generates session IDs.

Usage (CLI):
    python3 session_logger.py start [--model claude-opus-4-6]
    python3 session_logger.py status
    python3 session_logger.py end [--message "description"]

Usage (import):
    from session_logger import SessionLogger
    logger = SessionLogger(model="claude-opus-4-6")
    session_id = logger.start_session()
    # ... do work ...
    logger.end_session(message="what was done")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

LOG_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = LOG_DIR / "sessions"
ACTIVE_SESSION_FILE = SESSIONS_DIR / ".active_session"

# SECURITY: No credentials are hardcoded here. This service
# requires environment variables or a .env file.
# See .env.example for required variables.
load_dotenv(LOG_DIR.parent / ".env")


def _build_pg_dsn() -> str:
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


class SessionLogger:
    """Tracks an AI coding session."""

    def __init__(self, model: str = "claude-opus-4-6", session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.model = model
        self.started_at = datetime.now(UTC)
        self.ended_at: datetime | None = None
        self.timestamp_prefix = self.started_at.strftime("%Y-%m-%d_%H%M%S")
        self._json_path = SESSIONS_DIR / f"{self.timestamp_prefix}_session.json"
        self._md_path = SESSIONS_DIR / f"{self.timestamp_prefix}_summary.md"

    def start_session(self) -> str:
        """Initialize session, write initial JSON, return session_id."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Write active session pointer
        try:
            ACTIVE_SESSION_FILE.write_text(
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "timestamp_prefix": self.timestamp_prefix,
                        "model": self.model,
                        "started_at": self.started_at.isoformat(),
                    }
                )
            )
        except OSError as exc:
            logger.warning("Could not write active session file: %s", exc)

        # Write initial JSON
        self._write_json()

        # Record git HEAD at session start
        self._start_commit = self._get_git_head()

        return self.session_id

    def end_session(self, message: str | None = None) -> dict:
        """Finalize session: write summary markdown, insert DB record."""
        self.ended_at = datetime.now(UTC)
        self._write_json()
        self._write_markdown(message)
        db_ok = self._insert_to_db(message)

        # Clean up active session pointer
        try:
            if ACTIVE_SESSION_FILE.exists():
                ACTIVE_SESSION_FILE.unlink()
        except OSError:
            pass

        return {
            "session_id": self.session_id,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds(),
            "db_inserted": db_ok,
            "json_path": str(self._json_path),
            "md_path": str(self._md_path),
        }

    @classmethod
    def load_active(cls) -> SessionLogger | None:
        """Load the currently active session, or None."""
        if not ACTIVE_SESSION_FILE.exists():
            return None
        try:
            data = json.loads(ACTIVE_SESSION_FILE.read_text())
            s = cls(
                model=data.get("model", "unknown"),
                session_id=data.get("session_id"),
            )
            s.started_at = datetime.fromisoformat(data["started_at"])
            s.timestamp_prefix = data.get("timestamp_prefix", "unknown")
            s._json_path = SESSIONS_DIR / f"{s.timestamp_prefix}_session.json"
            s._md_path = SESSIONS_DIR / f"{s.timestamp_prefix}_summary.md"
            return s
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not load active session: %s", exc)
            return None

    def _write_json(self) -> None:
        """Write session state to JSON."""
        try:
            data = {
                "session_id": self.session_id,
                "model": self.model,
                "started_at": self.started_at.isoformat(),
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            }
            self._json_path.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.warning("Could not write session JSON: %s", exc)

    def _write_markdown(self, message: str | None) -> None:
        """Write human-readable session summary."""
        try:
            duration = (self.ended_at - self.started_at).total_seconds()

            # Get git changes during session
            git_diff = self._get_git_diff()
            git_log = self._get_git_log()

            lines = [
                f"# Session Summary: {self.timestamp_prefix}",
                "",
                "## Session Overview",
                f"- **Session ID:** `{self.session_id}`",
                f"- **Model:** {self.model}",
                f"- **Started:** {self.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"- **Ended:** {self.ended_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"- **Duration:** {duration:.0f}s ({duration / 60:.1f} min)",
                "",
            ]

            if message:
                lines += ["## Description", message, ""]

            if git_diff:
                lines += ["## Files Changed", "```", git_diff, "```", ""]

            if git_log:
                lines += ["## Git Commits During Session", "```", git_log, "```", ""]

            lines += [
                "## Errors Encountered",
                "_See terminal log for details._",
                "",
                "## AI Reasoning Captured",
                "_See terminal log for narrative context._",
                "",
            ]

            self._md_path.write_text("\n".join(lines) + "\n")
        except Exception as exc:
            logger.warning("Could not write markdown summary: %s", exc)

    def _get_git_head(self) -> str | None:
        """Get current git HEAD hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(LOG_DIR.parent),
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _get_git_diff(self) -> str:
        """Get git diff stat since session start."""
        try:
            start = getattr(self, "_start_commit", None)
            if start:
                cmd = ["git", "diff", "--stat", start, "HEAD"]
            else:
                cmd = ["git", "diff", "--stat", "HEAD~5", "HEAD"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(LOG_DIR.parent),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _get_git_log(self) -> str:
        """Get git log since session start."""
        try:
            start = getattr(self, "_start_commit", None)
            if start:
                cmd = ["git", "log", "--oneline", f"{start}..HEAD"]
            else:
                cmd = ["git", "log", "--oneline", "-10"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(LOG_DIR.parent),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _insert_to_db(self, message: str | None) -> bool:
        """Insert session summary into ai_activity_log. Returns True on success."""
        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 not installed — skipping DB insert")
            return False

        try:
            conn = psycopg2.connect(_build_pg_dsn())
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ai_activity_log (
                    session_id, session_start, session_end,
                    ai_model, action_type, action_timestamp,
                    reasoning, source, initiated_by
                ) VALUES (
                    %s::uuid, %s, %s,
                    %s, 'session_summary', %s,
                    %s, 'session_logger', 'claude-code'
                )
            """,
                (
                    self.session_id,
                    self.started_at,
                    self.ended_at,
                    self.model,
                    self.ended_at,
                    message or f"AI session {self.timestamp_prefix}",
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as exc:
            logger.warning("DB insert failed (non-fatal): %s", exc)
            return False


def main():
    parser = argparse.ArgumentParser(description="AI session logger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start a new session")
    p_start.add_argument("--model", default="claude-opus-4-6")

    sub.add_parser("status", help="Show active session info")

    p_end = sub.add_parser("end", help="End the active session")
    p_end.add_argument("--message", default=None, help="Session description")

    args = parser.parse_args()

    if args.command == "start":
        s = SessionLogger(model=args.model)
        sid = s.start_session()
        print(
            json.dumps(
                {
                    "session_id": sid,
                    "timestamp_prefix": s.timestamp_prefix,
                    "json_path": str(s._json_path),
                }
            )
        )

    elif args.command == "status":
        s = SessionLogger.load_active()
        if s is None:
            print('{"active": false}')
        else:
            elapsed = (datetime.now(UTC) - s.started_at).total_seconds()
            print(
                json.dumps(
                    {
                        "active": True,
                        "session_id": s.session_id,
                        "model": s.model,
                        "elapsed_seconds": round(elapsed),
                    },
                    indent=2,
                )
            )

    elif args.command == "end":
        s = SessionLogger.load_active()
        if s is None:
            print('{"error": "No active session"}', file=sys.stderr)
            sys.exit(1)
        result = s.end_session(message=args.message)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
