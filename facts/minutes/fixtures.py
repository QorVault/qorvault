"""Hard and advisory fixture checks for the minutes fact tables.

Hard fixtures must pass. They are never loosened to make a run go green: if one
fails, the run reports the failure and the diagnosis.

Run::

    .venv/bin/python fixtures.py
"""

from __future__ import annotations

import json
import re

import db

DISPOSITION_WORDS = {
    "adopted": r"carri|pass|adopt",
    "lost": r"fail|lost|died|not\s+carri",
    "tabled": r"tabl",
    "withdrawn": r"withdr",
}


def fixture_exec_sessions_2024() -> dict:
    """HARD: calendar 2024 should contain 27 executive sessions.

    Reported three ways because the corpus records executive sessions two ways
    and the expected figure's definition is not documented:

    * ``scheduled_meetings`` -- executive sessions convened as their own
      meeting, from the BoardDocs meeting census.
    * ``announcements`` -- executive sessions announced inside another
      meeting's minutes, one row per announcement (an extension counts
      separately, per the schema note).
    * A closing sentence ("The Executive Session was adjourned at 10:25 p.m.")
      is an end time for the session already open, not a new session. Counting
      closings as sessions would add exactly one to 2024.

    Returns:
        Check result with the counts under each definition.
    """
    row = db.query("""
        SELECT scheduled_meetings, announcements, total
        FROM facts.exec_sessions_by_year WHERE year = 2024
    """)
    scheduled, announced, total = row[0] if row else (0, 0, 0)
    closings = db.query("""
        SELECT count(*) FROM facts.executive_session e
        JOIN facts.meeting m USING (meeting_id)
        WHERE e.actual_end_time IS NOT NULL
          AND m.meeting_date >= '2024-01-01' AND m.meeting_date < '2025-01-01'
    """)[0][0]
    return {
        "name": "exec_sessions_2024",
        "severity": "HARD",
        "expected": 27,
        "actual": total,
        "passed": total == 27,
        "detail": {
            "scheduled_meetings": scheduled,
            "announcements": announced,
            "total": total,
            "closings_not_counted_as_sessions": closings,
            "total_if_closings_counted": total + closings,
        },
    }


def fixture_tally_within_attendance() -> dict:
    """HARD: yes+no+abstain must not exceed the directors present.

    Only motions that actually carry a tally are checked, and only for meetings
    where attendance was recorded -- a motion whose meeting has no attendance
    record cannot violate the constraint, it simply cannot be checked.

    Returns:
        Check result with any violating motions.
    """
    rows = db.query("""
        WITH present AS (
            SELECT meeting_id, count(*) AS n
            FROM facts.attendance
            WHERE status IN ('present', 'present_virtual', 'arrived_late',
                             'left_early')
            GROUP BY meeting_id
        )
        SELECT mo.motion_id, p.n,
               coalesce(mo.tally_yes,0) + coalesce(mo.tally_no,0)
                 + coalesce(mo.tally_abstain,0) AS cast_votes
        FROM facts.motion mo
        JOIN present p ON p.meeting_id = mo.meeting_id
        WHERE mo.tally_yes IS NOT NULL
           OR mo.tally_no IS NOT NULL
           OR mo.tally_abstain IS NOT NULL
    """)
    violations = [{"motion_id": r[0], "present": r[1], "cast": r[2]} for r in rows if r[2] > r[1]]
    return {
        "name": "tally_within_attendance",
        "severity": "HARD",
        "checked": len(rows),
        "violations": len(violations),
        "passed": not violations,
        "detail": {"examples": violations[:10]},
    }


def fixture_disposition_and_locator() -> dict:
    """HARD: every motion has a disposition and a locator proving it.

    The locator quote must contain a word consistent with the recorded
    disposition, so the citation resolves to text that actually shows the
    outcome rather than merely to the right document.

    Returns:
        Check result with any motions whose quote does not evidence the
        disposition.
    """
    rows = db.query("""
        SELECT motion_id, disposition, locator_document_id::text,
               locator_char_offset, coalesce(locator_quote,'')
        FROM facts.motion
    """)
    missing_locator = []
    quote_mismatch = []
    for motion_id, disposition, doc_id, offset, quote in rows:
        if not doc_id or offset is None or not quote:
            missing_locator.append(motion_id)
            continue
        pattern = DISPOSITION_WORDS.get(disposition)
        if pattern and not re.search(pattern, quote, re.I):
            quote_mismatch.append(
                {
                    "motion_id": motion_id,
                    "disposition": disposition,
                    "quote": quote[:120],
                }
            )
    return {
        "name": "disposition_and_locator",
        "severity": "HARD",
        "checked": len(rows),
        "missing_locator": len(missing_locator),
        "quote_does_not_evidence_disposition": len(quote_mismatch),
        "passed": not missing_locator and not quote_mismatch,
        "detail": {
            "missing_locator_examples": missing_locator[:10],
            "quote_mismatch_examples": quote_mismatch[:10],
        },
    }


def fixture_operator_hand_counts() -> dict:
    """HARD: six operator-chosen meetings must match a hand count.

    The hand counts have not been supplied, so this fixture cannot run. It is
    reported as BLOCKED rather than passed: a check that never executed is not
    a check that succeeded.

    Returns:
        Check result marked blocked.
    """
    return {
        "name": "operator_hand_counts",
        "severity": "HARD",
        "passed": None,
        "blocked": True,
        "detail": "Operator hand counts for six meetings (three per era) were "
        "not supplied before Phase 1. Cannot verify.",
    }


def advisory_vote_named_rate() -> dict:
    """ADVISORY: share of motions with named votes, by year.

    Returns:
        Per-year named-vote rates.
    """
    rows = db.query("""
        SELECT year, motions, motions_named, motions_unnamed, pct_unnamed,
               from_minutes, from_agenda_items
        FROM facts.votes_unnamed_by_year ORDER BY year
    """)
    return {
        "name": "vote_named_rate_by_year",
        "severity": "ADVISORY",
        "detail": [
            {
                "year": r[0],
                "motions": r[1],
                "named": r[2],
                "unnamed": r[3],
                "pct_unnamed": float(r[4]) if r[4] is not None else None,
                "from_minutes": r[5],
                "from_agenda_items": r[6],
            }
            for r in rows
        ],
    }


def advisory_parse_failures() -> dict:
    """ADVISORY: parse outcomes by year and status.

    Returns:
        Counts per status, and per-year document/motion totals.
    """
    by_status = db.query("""
        SELECT status, count(*) FROM facts.minutes_parse_log
        GROUP BY 1 ORDER BY 2 DESC
    """)
    by_year = db.query("""
        SELECT extract(year FROM minutes_date)::int AS yr,
               count(*) FILTER (WHERE status='parsed') AS parsed,
               count(*) FILTER (WHERE status='superseded_duplicate') AS superseded,
               count(*) FILTER (WHERE status NOT IN ('parsed','superseded_duplicate')) AS failed,
               sum(motions_found) AS motions,
               sum(exec_sessions_found) AS exec_sessions,
               sum(attendance_found) AS attendance
        FROM facts.minutes_parse_log
        WHERE minutes_date IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)
    return {
        "name": "parse_outcomes",
        "severity": "ADVISORY",
        "detail": {
            "by_status": {r[0]: r[1] for r in by_status},
            "by_year": [
                {
                    "year": r[0],
                    "parsed": r[1],
                    "superseded": r[2],
                    "failed": r[3],
                    "motions": int(r[4] or 0),
                    "exec_sessions": int(r[5] or 0),
                    "attendance": int(r[6] or 0),
                }
                for r in by_year
            ],
        },
    }


def main() -> int:
    """Run every fixture and print the results as JSON.

    Returns:
        0 when all hard fixtures pass or are blocked, 1 otherwise.
    """
    hard = [
        fixture_exec_sessions_2024(),
        fixture_tally_within_attendance(),
        fixture_disposition_and_locator(),
        fixture_operator_hand_counts(),
    ]
    advisory = [advisory_vote_named_rate(), advisory_parse_failures()]
    print(json.dumps({"hard": hard, "advisory": advisory}, indent=1, default=str))
    failed = [h for h in hard if h.get("passed") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
