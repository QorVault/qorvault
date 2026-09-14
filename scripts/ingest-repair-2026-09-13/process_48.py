#!/usr/bin/env python3
"""Run Stage 2 (document_processor) over the 48 unprocessed 2026-03-25 agenda items.

NOT EXECUTED. Written for operator review (Part 1); run only after approval.

Why this wrapper exists rather than calling document_processor directly:

  * document_processor selects work by ``processing_status = 'pending'`` and an
    optional ``--document-type``. It has no per-document filter. That is safe
    here only because the 48 target items are currently the *only* pending
    agenda items in the corpus -- this script asserts that precondition before
    running, and refuses to start if anything else is pending.
  * The task requires before/after counts and proof that no other document's
    rows changed. This captures a fingerprint of every non-target document
    before and after and compares them.

Extraction for ``agenda_item`` is ``strip_html(content_raw)``
(``document_processor/processor.py:209-213``) -- no OCR service and no file
access. The stale ``file_path`` values on these rows are therefore irrelevant,
and the stopped OCR service is not a blocker.

Every SQL statement below is a fixed literal with no interpolation of any kind,
per the project rule that SQL is never built by string formatting. The target
meetings are hard-coded rather than passed in, because this script exists to
repair one specific, already-reviewed incident and must not be repointed at
arbitrary rows.

Credentials are read from the environment at runtime (POSTGRES_* or
DATABASE_URL). This script does not read, write, or modify .env.

Usage:
    ./process_48.py --dry-run     # report only, no processing
    ./process_48.py               # run Stage 2, with before/after verification
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The three 2026-03-25 meetings whose agenda items Stage 2 never processed.
EXPECTED_COUNT = 48

# --- Fixed SQL. No interpolation, no user input, no composition. -------------

SQL_TARGET_PENDING = """
SELECT count(*) FROM documents d
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D')
  AND d.processing_status = 'pending';
"""

SQL_TARGET_COMPLETE = """
SELECT count(*) FROM documents d
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D')
  AND d.processing_status = 'complete';
"""

SQL_TARGET_FAILED = """
SELECT count(*) FROM documents d
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D')
  AND d.processing_status IN ('failed','deferred');
"""

SQL_TARGET_CHUNKS = """
SELECT count(*) FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D');
"""

SQL_TARGET_WITH_CHUNKS = """
SELECT count(*) FROM documents d
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D')
  AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id);
"""

# Blast radius: fingerprint of every document that is NOT a target. Must be
# byte-identical before and after the run.
SQL_OTHER_FINGERPRINT = """
SELECT md5(string_agg(d.id::text || ':' || d.processing_status || ':' ||
                      d.updated_at::text, '|' ORDER BY d.id))
FROM documents d
WHERE NOT (d.document_type = 'agenda_item'
           AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D'));
"""

SQL_OTHER_CHUNKS = """
SELECT count(*) FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE NOT (d.document_type = 'agenda_item'
           AND d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D'));
"""

# Any pending agenda item outside the three target meetings would be swept into
# the same type-scoped processor run.
SQL_STRAY_PENDING = """
SELECT count(*) FROM documents d
WHERE d.processing_status = 'pending'
  AND d.document_type = 'agenda_item'
  AND NOT (d.meeting_id IN ('DS4MSK5CA5C5','DS4MVC5CCBE6','DSCM9K5A287D'));
"""


def psql(sql: str) -> str:
    """Run one SQL statement in the corpus container, returning raw output.

    Args:
        sql: A fixed SQL literal from this module. Never a constructed string.

    Returns:
        The trimmed stdout of psql in unaligned, tuples-only mode.
    """
    return subprocess.run(
        [
            "podman",
            "exec",
            "-i",
            "boarddocs-postgres",
            "psql",
            "-U",
            "boarddocs",
            "-d",
            "boarddocs",
            "-P",
            "pager=off",
            "-At",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def snapshot() -> dict[str, str]:
    """Capture target-set counts plus a fingerprint of every other document.

    Returns:
        Mapping of measure name to its value as returned by PostgreSQL.
    """
    return {
        "target_pending": psql(SQL_TARGET_PENDING),
        "target_complete": psql(SQL_TARGET_COMPLETE),
        "target_failed": psql(SQL_TARGET_FAILED),
        "target_chunks": psql(SQL_TARGET_CHUNKS),
        "target_with_chunks": psql(SQL_TARGET_WITH_CHUNKS),
        "other_fingerprint": psql(SQL_OTHER_FINGERPRINT),
        "other_chunks": psql(SQL_OTHER_CHUNKS),
    }


def show(label: str, snap: dict[str, str]) -> None:
    """Print a labelled snapshot as an aligned block.

    Args:
        label: Heading for the block, e.g. ``BEFORE``.
        snap: Snapshot mapping as returned by :func:`snapshot`.
    """
    print(f"\n=== {label} ===")
    for key, value in snap.items():
        print(f"  {key:<20} {value}")


def preflight() -> int:
    """Refuse to run unless the corpus matches what was approved.

    Returns:
        0 if it is safe to run Stage 2, 1 otherwise.
    """
    problems = []

    target_pending = int(psql(SQL_TARGET_PENDING))
    if target_pending != EXPECTED_COUNT:
        problems.append(
            f"expected {EXPECTED_COUNT} pending agenda items in the target " f"meetings, found {target_pending}"
        )

    stray = int(psql(SQL_STRAY_PENDING))
    if stray:
        problems.append(
            f"{stray} pending agenda_item(s) outside the target meetings; a "
            "type-scoped processor run would also process those"
        )

    if problems:
        print("PREFLIGHT FAILED — not running Stage 2:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"preflight OK: exactly {EXPECTED_COUNT} pending agenda items, " "no stray pending agenda items elsewhere")
    return 0


def verify(before: dict[str, str], after: dict[str, str], dry_run: bool) -> bool:
    """Compare before/after snapshots and report each pass or fail.

    Args:
        before: Snapshot taken before Stage 2 ran.
        after: Snapshot taken after Stage 2 ran.
        dry_run: True when the processor ran in its own dry-run mode, in which
            case no state change is expected.

    Returns:
        True if every check passed.
    """
    print("\n=== VERIFICATION ===")
    ok = True

    if dry_run:
        print("  dry run — no state change expected")
    else:
        if after["target_pending"] != "0":
            print(f"  FAIL: {after['target_pending']} target items still pending")
            ok = False
        else:
            print("  PASS: 0 target items still pending")

        if after["target_complete"] != str(EXPECTED_COUNT):
            print(f"  FAIL: {after['target_complete']}/{EXPECTED_COUNT} marked complete")
            ok = False
        else:
            print(f"  PASS: all {EXPECTED_COUNT} marked complete")

        if after["target_failed"] != "0":
            print(f"  FAIL: {after['target_failed']} target items failed/deferred")
            ok = False
        else:
            print("  PASS: 0 failed or deferred")

        if after["target_with_chunks"] != str(EXPECTED_COUNT):
            print(f"  FAIL: only {after['target_with_chunks']}/{EXPECTED_COUNT} " "documents have chunk rows")
            ok = False
        else:
            print(f"  PASS: all {EXPECTED_COUNT} documents have chunk rows " f"({after['target_chunks']} chunks total)")

    # These two must hold in every mode: nothing outside the 48 may change.
    if before["other_fingerprint"] != after["other_fingerprint"]:
        print("  FAIL: other documents changed — fingerprint differs")
        ok = False
    else:
        print("  PASS: no other document row changed")

    if before["other_chunks"] != after["other_chunks"]:
        print(
            f"  FAIL: chunk count for other documents changed " f"({before['other_chunks']} -> {after['other_chunks']})"
        )
        ok = False
    else:
        print("  PASS: no chunk added or removed for any other document")

    return ok


def main() -> int:
    """Entry point: preflight, snapshot, run Stage 2, snapshot, verify.

    Returns:
        Process exit code -- 0 on success, 1 on any failed check.
    """
    parser = argparse.ArgumentParser(description="Stage 2 runner for the 48 agenda items")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report counts and run the processor in its own dry-run mode",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if preflight() != 0:
        return 1

    before = snapshot()
    show("BEFORE", before)

    cmd = [
        sys.executable,
        "-m",
        "document_processor",
        "--document-type",
        "agenda_item",
        "--workers",
        str(args.workers),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"\n=== running Stage 2: {' '.join(cmd)} ===")
    # Credentials come from the ambient environment; .env is not touched.
    result = subprocess.run(cmd, cwd=REPO / "document_processor", env=os.environ.copy())
    print(f"=== processor exit code: {result.returncode} ===")

    after = snapshot()
    show("AFTER", after)

    ok = verify(before, after, args.dry_run)

    print(
        "\nNOTE: embeddings are Stage 3 (embedding_pipeline) and are NOT run "
        "here. Chunks will read embedding_status='pending' until that stage "
        "runs; that is expected, not a failure."
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
