#!/usr/bin/env python3
"""Session finalizer: reads terminal log, extracts AI activity data,
generates markdown summary, and inserts session record into PostgreSQL.

Called automatically by the claude-session shell wrapper on exit.

Usage:
    python3 finalize_session.py --session-id UUID --log-file path/to/terminal.log
    python3 finalize_session.py --session-id UUID --log-file path/to/terminal.log --start-commit abc123
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

LOG_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = LOG_DIR / "sessions"
PROJECT_ROOT = LOG_DIR.parent

# SECURITY: No credentials are hardcoded here. This service
# requires environment variables or a .env file.
# See .env.example for required variables.
load_dotenv(PROJECT_ROOT / ".env")


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


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def parse_terminal_log(log_path: Path) -> dict:
    """Extract structured data from a terminal session log."""
    result = {
        "model_mentions": set(),
        "file_paths": set(),
        "commands": [],
        "errors": [],
        "reasoning": [],
    }

    if not log_path.exists():
        logger.warning("Terminal log not found: %s", log_path)
        return {k: list(v) if isinstance(v, set) else v for k, v in result.items()}

    try:
        content = log_path.read_text(errors="replace")
    except OSError as exc:
        logger.warning("Could not read terminal log: %s", exc)
        return {k: list(v) if isinstance(v, set) else v for k, v in result.items()}

    content = strip_ansi(content)

    # Detect model mentions
    model_patterns = [
        r"claude-opus-4-6",
        r"claude-sonnet-4-6",
        r"claude-haiku-4-5",
        r"claude-opus-4-20250514",
        r"claude-sonnet-4-20250514",
        r"gpt-4[o\-]*\w*",
        r"qwen\S+",
    ]
    for pattern in model_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            result["model_mentions"].add(match.group(0))

    # Extract file paths (absolute paths starting with /home or relative ./ paths)
    for match in re.finditer(r"(/home/qorvault/projects/ksd-boarddocs-rag/\S+|\.\/\S+)", content):
        path = match.group(0).rstrip(".,;:)'\"")
        if any(
            path.endswith(ext)
            for ext in [
                ".py",
                ".sh",
                ".sql",
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".cfg",
                ".service",
                ".rules",
                ".html",
                ".css",
                ".js",
                ".ts",
                ".jsonl",
            ]
        ):
            result["file_paths"].add(path)

    # Extract bash/shell commands (lines starting with $ or common command prefixes)
    for match in re.finditer(r"(?:^|\n)\s*\$\s+(.+?)(?:\n|$)", content):
        cmd = match.group(1).strip()
        if len(cmd) > 5 and len(cmd) < 500:
            result["commands"].append(cmd)

    # Extract errors
    for match in re.finditer(
        r"(?:Error|ERROR|Traceback|FAILED|Exception|fatal).*?(?:\n|$)",
        content,
        re.IGNORECASE,
    ):
        err = match.group(0).strip()
        if len(err) > 5:
            result["errors"].append(err[:500])

    # Extract AI reasoning (narrative patterns)
    reasoning_patterns = [
        r"(?:Let me|I'll|I need to|I should|I'm going to|First,|Next,|Now |The reason).*?(?:\n|$)",
    ]
    for pattern in reasoning_patterns:
        for match in re.finditer(pattern, content):
            snippet = match.group(0).strip()
            if len(snippet) > 20 and len(snippet) < 500:
                result["reasoning"].append(snippet)

    # Deduplicate and convert sets to lists
    result["model_mentions"] = sorted(result["model_mentions"])
    result["file_paths"] = sorted(result["file_paths"])
    result["errors"] = result["errors"][:50]  # Cap at 50
    result["reasoning"] = result["reasoning"][:100]  # Cap at 100

    return result


def get_git_diff(start_commit: str | None) -> str:
    """Get git diff stat between session start and current HEAD."""
    try:
        if start_commit:
            cmd = ["git", "diff", "--stat", start_commit, "HEAD"]
        else:
            cmd = ["git", "diff", "--stat", "HEAD~10", "HEAD"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_git_log(start_commit: str | None) -> str:
    """Get git log since session start."""
    try:
        if start_commit:
            cmd = ["git", "log", "--oneline", f"{start_commit}..HEAD"]
        else:
            cmd = ["git", "log", "--oneline", "-10"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_current_head() -> str | None:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def generate_markdown(
    session_id: str,
    session_start: datetime,
    session_end: datetime,
    parsed: dict,
    git_diff: str,
    git_log: str,
    start_commit: str | None,
    end_commit: str | None,
) -> str:
    """Generate markdown summary of the session."""
    duration = (session_end - session_start).total_seconds()
    model = parsed["model_mentions"][0] if parsed["model_mentions"] else "unknown"

    lines = [
        "# Session Summary",
        "",
        "## Session Overview",
        f"- **Session ID:** `{session_id}`",
        f"- **Model detected:** {model}",
        f"- **Started:** {session_start.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Ended:** {session_end.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Duration:** {duration:.0f}s ({duration / 60:.1f} min)",
    ]
    if start_commit:
        lines.append(f"- **Start commit:** `{start_commit[:8]}`")
    if end_commit:
        lines.append(f"- **End commit:** `{end_commit[:8]}`")
    lines.append("")

    # Files Changed
    lines.append("## Files Changed")
    if git_diff:
        lines += ["```", git_diff, "```"]
    elif parsed["file_paths"]:
        for fp in parsed["file_paths"]:
            lines.append(f"- `{fp}`")
    else:
        lines.append("_No file changes detected._")
    lines.append("")

    # Commands Executed
    lines.append("## Commands Executed")
    if parsed["commands"]:
        for cmd in parsed["commands"][:30]:
            lines.append(f"- `{cmd}`")
    else:
        lines.append("_No commands extracted from terminal log._")
    lines.append("")

    # Errors Encountered
    lines.append("## Errors Encountered")
    if parsed["errors"]:
        for err in parsed["errors"][:20]:
            lines.append(f"- {err}")
    else:
        lines.append("_No errors detected._")
    lines.append("")

    # AI Reasoning Captured
    lines.append("## AI Reasoning Captured")
    if parsed["reasoning"]:
        for r in parsed["reasoning"][:20]:
            lines.append(f"> {r}")
            lines.append("")
    else:
        lines.append("_No reasoning snippets extracted._")
    lines.append("")

    # Git Commits
    lines.append("## Git Commits Made During Session")
    if git_log:
        lines += ["```", git_log, "```"]
    else:
        lines.append("_No commits during this session._")
    lines.append("")

    return "\n".join(lines) + "\n"


def insert_session_to_db(
    session_id: str,
    session_start: datetime,
    session_end: datetime,
    parsed: dict,
    end_commit: str | None,
) -> bool:
    """Insert session summary into ai_activity_log."""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed — skipping DB insert")
        return False

    try:
        model = parsed["model_mentions"][0] if parsed["model_mentions"] else "unknown"
        conn = psycopg2.connect(_build_pg_dsn())
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_activity_log (
                session_id, session_start, session_end,
                ai_model, action_type, action_timestamp,
                git_commit_hash, source, initiated_by,
                raw_event
            ) VALUES (
                %s::uuid, %s, %s,
                %s, 'session_complete', %s,
                %s, 'session_finalizer', 'claude-session',
                %s::jsonb
            )
        """,
            (
                session_id,
                session_start,
                session_end,
                model,
                session_end,
                end_commit,
                json.dumps(
                    {
                        "files_touched": len(parsed["file_paths"]),
                        "commands_run": len(parsed["commands"]),
                        "errors": len(parsed["errors"]),
                        "models": parsed["model_mentions"],
                    }
                ),
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
    parser = argparse.ArgumentParser(description="Finalize a Claude Code session")
    parser.add_argument("--session-id", required=True, help="Session UUID")
    parser.add_argument("--log-file", required=True, help="Path to terminal log")
    parser.add_argument("--start-commit", default=None, help="Git commit hash at session start")
    parser.add_argument("--session-start", default=None, help="ISO 8601 session start time")
    args = parser.parse_args()

    session_id = args.session_id
    log_file = Path(args.log_file)

    # Determine session timing
    if args.session_start:
        session_start = datetime.fromisoformat(args.session_start)
    elif log_file.exists():
        # Use file creation time as proxy
        stat = log_file.stat()
        session_start = datetime.fromtimestamp(stat.st_ctime, tz=UTC)
    else:
        session_start = datetime.now(UTC)

    session_end = datetime.now(UTC)

    logger.info("Finalizing session %s", session_id)
    logger.info("Terminal log: %s", log_file)

    # Parse terminal log
    parsed = parse_terminal_log(log_file)
    logger.info(
        "Parsed: %d models, %d files, %d commands, %d errors, %d reasoning",
        len(parsed["model_mentions"]),
        len(parsed["file_paths"]),
        len(parsed["commands"]),
        len(parsed["errors"]),
        len(parsed["reasoning"]),
    )

    # Git data
    git_diff = get_git_diff(args.start_commit)
    git_log = get_git_log(args.start_commit)
    end_commit = get_current_head()

    # Generate markdown
    timestamp_prefix = session_start.strftime("%Y-%m-%d_%H%M%S")
    md_path = SESSIONS_DIR / f"{timestamp_prefix}_summary.md"
    md_content = generate_markdown(
        session_id,
        session_start,
        session_end,
        parsed,
        git_diff,
        git_log,
        args.start_commit,
        end_commit,
    )
    try:
        md_path.write_text(md_content)
        logger.info("Summary written to %s", md_path)
    except OSError as exc:
        logger.error("Could not write summary: %s", exc)

    # Insert to DB
    db_ok = insert_session_to_db(
        session_id,
        session_start,
        session_end,
        parsed,
        end_commit,
    )
    logger.info("DB insert: %s", "success" if db_ok else "failed")

    # Print result
    print(
        json.dumps(
            {
                "session_id": session_id,
                "summary_path": str(md_path),
                "db_inserted": db_ok,
                "files_touched": len(parsed["file_paths"]),
                "commands_found": len(parsed["commands"]),
                "errors_found": len(parsed["errors"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
