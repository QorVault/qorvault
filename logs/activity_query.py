#!/usr/bin/env python3
"""CLI tool for querying AI activity logs.

Provides a simple interface for reviewing activity stored in the
ai_activity_log PostgreSQL table.

Usage:
    activity_query sessions --last 10
    activity_query session SESSION_ID
    activity_query files --path "*.py" --since 2026-02-27
    activity_query changes --since 2026-02-27 --until 2026-02-28
    activity_query who-did /home/qorvault/projects/ksd-boarddocs-rag/logs/aggregator.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# SECURITY: No credentials are hardcoded here. This service
# requires environment variables or a .env file.
# See .env.example for required variables.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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


def get_connection():
    """Get a psycopg2 connection."""
    import psycopg2

    return psycopg2.connect(_build_pg_dsn())


def cmd_sessions(args):
    """List last N sessions."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT ON (session_id)
            session_id, session_start, session_end,
            ai_model, action_type, source
        FROM ai_activity_log
        WHERE action_type IN ('session_summary', 'session_complete')
        ORDER BY session_id, session_start DESC
        LIMIT %s
    """,
        (args.last,),
    )

    rows = cur.fetchall()
    if not rows:
        # Fall back to showing any recent activity
        cur.execute(
            """
            SELECT session_id, MIN(action_timestamp), MAX(action_timestamp),
                   ai_model, COUNT(*), string_agg(DISTINCT source, ', ')
            FROM ai_activity_log
            GROUP BY session_id, ai_model
            ORDER BY MIN(action_timestamp) DESC
            LIMIT %s
        """,
            (args.last,),
        )
        rows = cur.fetchall()
        if not rows:
            print("No sessions found.")
            return

        print(f"{'Session ID':<38} {'Start':<22} {'End':<22} {'Model':<20} {'Actions':<8} {'Sources'}")
        print("-" * 130)
        for row in rows:
            sid = str(row[0])[:36]
            start = row[1].strftime("%Y-%m-%d %H:%M") if row[1] else "?"
            end = row[2].strftime("%Y-%m-%d %H:%M") if row[2] else "?"
            model = str(row[3] or "?")[:18]
            actions = row[4]
            sources = str(row[5] or "?")
            print(f"{sid:<38} {start:<22} {end:<22} {model:<20} {actions:<8} {sources}")
    else:
        print(f"{'Session ID':<38} {'Start':<22} {'End':<22} {'Model':<20} {'Source'}")
        print("-" * 120)
        for row in rows:
            sid = str(row[0])[:36]
            start = row[1].strftime("%Y-%m-%d %H:%M") if row[1] else "?"
            end = row[2].strftime("%Y-%m-%d %H:%M") if row[2] else "?"
            model = str(row[3] or "?")[:18]
            source = str(row[5] or "?")
            print(f"{sid:<38} {start:<22} {end:<22} {model:<20} {source}")

    cur.close()
    conn.close()


def cmd_session(args):
    """Full details of a specific session."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, session_id, session_start, session_end,
               ai_model, action_type, action_timestamp,
               file_path, command_executed, reasoning,
               outcome, source, success, error_message,
               raw_event
        FROM ai_activity_log
        WHERE session_id = %s::uuid
        ORDER BY action_timestamp
    """,
        (args.session_id,),
    )

    rows = cur.fetchall()
    if not rows:
        print(f"No records found for session {args.session_id}")
        return

    print(f"Session: {args.session_id}")
    print(f"Records: {len(rows)}")
    print()

    for row in rows:
        print(
            f"  [{row[6].strftime('%H:%M:%S') if row[6] else '?'}] "
            f"{row[5]:<25} "
            f"source={row[11] or '?':<15} "
            f"file={row[7] or '-'}"
        )
        if row[8]:  # command
            print(f"    cmd: {row[8][:80]}")
        if row[9]:  # reasoning
            print(f"    why: {row[9][:80]}")
        if row[13]:  # error
            print(f"    err: {row[13][:80]}")
        print()

    cur.close()
    conn.close()


def cmd_files(args):
    """All activity touching files matching a pattern."""
    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT action_timestamp, action_type, file_path,
               source, session_id, ai_model
        FROM ai_activity_log
        WHERE file_path LIKE %s
    """
    params = [f"%{args.path}%"]

    if args.since:
        query += " AND action_timestamp >= %s"
        params.append(args.since)

    query += " ORDER BY action_timestamp DESC LIMIT 100"

    cur.execute(query, params)
    rows = cur.fetchall()

    if not rows:
        print(f"No activity found for files matching '{args.path}'")
        return

    print(f"{'Timestamp':<22} {'Type':<25} {'Source':<15} {'File Path'}")
    print("-" * 120)
    for row in rows:
        ts = row[0].strftime("%Y-%m-%d %H:%M:%S") if row[0] else "?"
        print(f"{ts:<22} {row[1]:<25} {row[3] or '?':<15} {row[2] or '-'}")

    print(f"\n{len(rows)} records found.")
    cur.close()
    conn.close()


def cmd_changes(args):
    """All file changes in date range."""
    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT action_timestamp, action_type, file_path,
               git_commit_hash, source, session_id
        FROM ai_activity_log
        WHERE action_timestamp >= %s
    """
    params = [args.since]

    if args.until:
        query += " AND action_timestamp <= %s"
        params.append(args.until)

    query += " AND file_path IS NOT NULL ORDER BY action_timestamp"

    cur.execute(query, params)
    rows = cur.fetchall()

    if not rows:
        print(f"No file changes found since {args.since}")
        return

    print(f"{'Timestamp':<22} {'Type':<25} {'Commit':<10} {'File Path'}")
    print("-" * 120)
    for row in rows:
        ts = row[0].strftime("%Y-%m-%d %H:%M:%S") if row[0] else "?"
        commit = (row[3] or "-")[:8]
        print(f"{ts:<22} {row[1]:<25} {commit:<10} {row[2] or '-'}")

    print(f"\n{len(rows)} changes found.")
    cur.close()
    conn.close()


def cmd_who_did(args):
    """Complete history of all AI activity touching a specific file."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT action_timestamp, action_type, source,
               ai_model, session_id, command_executed,
               reasoning, outcome, git_commit_hash
        FROM ai_activity_log
        WHERE file_path LIKE %s
        ORDER BY action_timestamp
    """,
        (f"%{args.file_path}%",),
    )

    rows = cur.fetchall()

    if not rows:
        print(f"No activity found for: {args.file_path}")
        return

    print(f"Activity history for: {args.file_path}")
    print(f"Total records: {len(rows)}")
    print()

    for row in rows:
        ts = row[0].strftime("%Y-%m-%d %H:%M:%S") if row[0] else "?"
        print(f"  [{ts}] {row[1]} (source: {row[2] or '?'})")
        if row[3]:
            print(f"    model: {row[3]}")
        if row[5]:
            print(f"    cmd: {row[5][:80]}")
        if row[6]:
            print(f"    why: {row[6][:80]}")
        if row[8]:
            print(f"    commit: {row[8][:8]}")
        print()

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        prog="activity_query",
        description="Query AI activity logs for the KSD BoardDocs RAG project",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sessions
    p_sessions = sub.add_parser("sessions", help="List recent sessions")
    p_sessions.add_argument("--last", type=int, default=10, help="Number of sessions")

    # session
    p_session = sub.add_parser("session", help="Full session details")
    p_session.add_argument("session_id", help="Session UUID")

    # files
    p_files = sub.add_parser("files", help="Activity on files matching pattern")
    p_files.add_argument("--path", required=True, help="File path pattern")
    p_files.add_argument("--since", default=None, help="Since date (YYYY-MM-DD)")

    # changes
    p_changes = sub.add_parser("changes", help="File changes in date range")
    p_changes.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    p_changes.add_argument("--until", default=None, help="End date (YYYY-MM-DD)")

    # who-did
    p_who = sub.add_parser("who-did", help="Who touched this file?")
    p_who.add_argument("file_path", help="File path to query")

    args = parser.parse_args()

    try:
        if args.command == "sessions":
            cmd_sessions(args)
        elif args.command == "session":
            cmd_session(args)
        elif args.command == "files":
            cmd_files(args)
        elif args.command == "changes":
            cmd_changes(args)
        elif args.command == "who-did":
            cmd_who_did(args)
    except ImportError:
        print("Error: psycopg2 is required. Install with: pip install psycopg2", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
