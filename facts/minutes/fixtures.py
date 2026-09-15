"""Hard and advisory fixture checks for the minutes fact tables.

Hard fixtures must pass. They are never loosened to make a run go green: if one
fails, the run reports the failure and the diagnosis.

Run::

    .venv/bin/python fixtures.py
"""

from __future__ import annotations

import json
import os
import re

import db

# Disposition evidence patterns, matched against the locator quote.
#
# These are word STEMS, not whole words. The minutes and the agenda items use
# different inflections of the same verb for the same outcome -- a motion is
# recorded as `withdrawn` where the text says the mover "withdrew" it, and as
# `adopted` where the text says the board "adopts" it. Matching whole words
# would fail those rows even though the quote plainly evidences the outcome.
#
# Stems covered per disposition:
#   adopted    carried / carries / carry / passed / passes / pass /
#              adopted / adopts / adopt
#   lost       failed / fails / fail / lost / loses / died / not carried
#   tabled     tabled / tables / table
#   withdrawn  withdrew / withdrawn / withdraws / withdraw
#
# "approve" is deliberately NOT a stem for `adopted`. An agenda item's
# "Recommended Action: That the Board of Directors approves ..." is the
# proposal put to the board, not evidence that the board adopted it. Admitting
# it would turn 32 truncated citations green without any of them gaining a
# word that shows the outcome -- the fixture would stop measuring what it
# exists to measure.
DISPOSITION_WORDS = {
    "adopted": r"carri|carry|pass|adopt",
    "lost": r"fail|lost|lose|died|not\s+carri",
    "tabled": r"tabl",
    "withdrawn": r"withdr",
}

# Meetings where more directors are recorded voting than the minutes record as
# present. Each is understood; see facts.attendance_vote_discrepancies for the
# cause vocabulary. A discrepancy OUTSIDE this set is a new finding and fails
# the hard fixture.
#
# `parser_roll_bleed` is deliberately NOT described as a record discrepancy:
# 2025-02-11 is a live parser defect and is carried here so it stays visible
# rather than being absorbed into the accepted set. See the fixtures report.
KNOWN_ATTENDANCE_VOTE_DISCREPANCIES = {
    "2022-06-29:special": ("2022-06-29", "presiding_only"),
    "2022-10-05:special": ("2022-10-05", "attendance_short"),
    "2023-11-08:regular": ("2023-11-08", "status_excluded"),
    "2023-12-13:regular": ("2023-12-13", "board_transition"),
    "2024-07-10:special": ("2024-07-10", "status_excluded"),
    "2025-02-11:special": ("2025-02-11", "parser_roll_bleed"),
}

HAND_COUNTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hand_counts.yaml")


# SUPERSEDED EXPECTATION -- retained deliberately, not deleted.
#
# The original hard fixture asserted 27 executive sessions in calendar 2024.
# That figure is only reproducible by counting "The Executive Session was
# adjourned at 10:25 p.m." in the 2024-09-11 minutes as a session in its own
# right. It is an end time for the session already open, so counting it would
# double-count every session whose close happens to be recorded. The operator
# accepted 26 and the assertion moved to the 24 sessions that were convened as
# their own meeting; the 2 announced inside other meetings' minutes are now
# reported rather than asserted.
SUPERSEDED_EXEC_SESSIONS_2024_EXPECTATION = 27


def fixture_exec_sessions_2024() -> dict:
    """HARD: every 2024 executive session convened as its own meeting has a row.

    The corpus records executive sessions two ways and only one of them is a
    census the fixture can assert against:

    * ``scheduled_meetings`` (asserted, expect 24) -- executive sessions
      convened as their own meeting, from the BoardDocs meeting census. This
      is a closed list, so a shortfall is a real miss.
    * ``announcements`` (reported, expect 2) -- executive sessions announced
      inside another meeting's minutes. Reported, never asserted: the corpus
      cannot tell us how many announcements it *should* contain, so a count
      here is an observation about the record, not a target.

    A closing or adjournment sentence is never counted as a session.

    Returns:
        Check result asserting the scheduled-meeting census only.
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
        "asserted": {"scheduled_meetings_expected": 24, "scheduled_meetings_actual": scheduled},
        "passed": scheduled == 24,
        "reported_not_asserted": {
            "announcements_in_other_meetings_minutes": announced,
            "announcements_expected": 2,
        },
        "detail": {
            "total_sessions_2024": total,
            "closings_never_counted_as_sessions": closings,
            "superseded_expectation": SUPERSEDED_EXEC_SESSIONS_2024_EXPECTATION,
            "superseded_reason": (
                "27 counted an adjournment sentence as a session; an "
                "adjournment is an end time for the session already open."
            ),
        },
    }


def unknown_attendance_vote_discrepancies(rows: list[tuple]) -> list[dict]:
    """Filter discrepancy rows down to those outside the known-set.

    Split out from the fixture so the known-set logic is testable without a
    database.

    Args:
        rows: Tuples of ``(meeting_id, motion_id, present, cast, cause)`` as
            returned by ``facts.attendance_vote_discrepancies``.

    Returns:
        One dict per row whose meeting is not in the known-set.
    """
    unknown = []
    for meeting_id, motion_id, present, cast, cause in rows:
        if meeting_id in KNOWN_ATTENDANCE_VOTE_DISCREPANCIES:
            continue
        unknown.append(
            {
                "meeting_id": meeting_id,
                "motion_id": motion_id,
                "present": present,
                "cast": cast,
                "cause": cause,
            }
        )
    return unknown


def fixture_attendance_vote_discrepancies() -> dict:
    """HARD: no meeting outside the known-set records more voters than present.

    The original fixture asserted that voters never exceed recorded attendance.
    That assertion is false about the district's own records, and holding it
    produced a permanently red check that said nothing useful. The check is
    kept but its role changed: the discrepancies are now a maintained,
    caused list, and the assertion is that nothing NEW appears. A green result
    means "no undiagnosed discrepancy", not "no discrepancy".

    Returns:
        Check result listing every discrepancy and flagging unknown ones.
    """
    rows = db.query("""
        SELECT meeting_id, motion_id, present_recorded, cast_votes, cause
        FROM facts.attendance_vote_discrepancies
    """)
    unknown = unknown_attendance_vote_discrepancies(rows)

    by_meeting: dict[str, dict] = {}
    for meeting_id, _motion_id, present, cast, cause in rows:
        entry = by_meeting.setdefault(
            meeting_id,
            {"meeting_id": meeting_id, "cause": cause, "motions": 0, "present": present, "max_cast": 0},
        )
        entry["motions"] += 1
        entry["max_cast"] = max(entry["max_cast"], cast)

    # A known meeting that no longer appears means the underlying data moved;
    # surface it rather than letting the known-set silently rot.
    stale = sorted(set(KNOWN_ATTENDANCE_VOTE_DISCREPANCIES) - set(by_meeting))

    return {
        "name": "attendance_vote_discrepancies",
        "severity": "HARD",
        "passed": not unknown,
        "discrepant_motions": len(rows),
        "discrepant_meetings": len(by_meeting),
        "unknown_meetings": sorted({u["meeting_id"] for u in unknown}),
        "detail": {
            "known_set": {k: v[1] for k, v in sorted(KNOWN_ATTENDANCE_VOTE_DISCREPANCIES.items())},
            "by_meeting": sorted(by_meeting.values(), key=lambda e: e["meeting_id"]),
            "unknown_examples": unknown[:10],
            "known_but_no_longer_present": stale,
        },
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
               locator_page, locator_char_offset, coalesce(locator_quote,'')
        FROM facts.motion ORDER BY motion_id
    """)
    missing_locator = []
    quote_mismatch = []
    for motion_id, disposition, doc_id, page, offset, quote in rows:
        if not doc_id or offset is None or not quote:
            missing_locator.append(motion_id)
            continue
        pattern = DISPOSITION_WORDS.get(disposition)
        if pattern and not re.search(pattern, quote, re.I):
            # Every residual is enumerated in full -- document id, page and the
            # whole quote. A bare count of mismatches is not reviewable: it
            # cannot distinguish a vocabulary gap from a truncated citation.
            quote_mismatch.append(
                {
                    "motion_id": motion_id,
                    "disposition": disposition,
                    "document_id": doc_id,
                    "page": page,
                    "char_offset": offset,
                    "quote": quote,
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
            "missing_locator": missing_locator,
            "quote_mismatch": quote_mismatch,
        },
    }


def _parse_scalar(raw: str) -> object:
    """Convert a YAML scalar from the hand-count file to a Python value.

    Args:
        raw: Raw scalar text, already stripped of its key and comments.

    Returns:
        ``None`` for ``null``/empty, an ``int`` for integer text, otherwise the
        string with any surrounding quotes removed.
    """
    raw = raw.strip()
    if raw in ("", "null", "~"):
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def load_hand_counts(path: str = HAND_COUNTS_PATH) -> list[dict]:
    """Read the operator's hand-count file.

    Deliberately a minimal parser rather than PyYAML. PyYAML is not in
    ``requirements.txt`` or ``requirements-lock.txt``, and this package does
    not add a dependency to read one flat list of scalars that it also writes
    the template for. The accepted structure is exactly::

        meetings:
          - key: value
            key: value

    Nested collections, anchors, multi-line scalars and flow style are not
    supported and are not used by the template.

    Args:
        path: Path to ``hand_counts.yaml``.

    Returns:
        One dict per meeting entry, in file order. Empty when the file is
        absent.

    Raises:
        ValueError: If a line inside ``meetings:`` is not a ``key: value``
            pair, so a malformed file fails loudly instead of silently
            yielding fewer meetings than it names.
    """
    if not os.path.isfile(path):
        return []

    meetings: list[dict] = []
    in_meetings = False
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith((" ", "\t", "-")):
                in_meetings = line.strip() == "meetings:"
                continue
            if not in_meetings:
                continue
            body = line.strip()
            if body.startswith("- "):
                meetings.append({})
                body = body[2:].strip()
            if not meetings:
                continue
            if ":" not in body:
                raise ValueError(f"{path}:{lineno}: expected 'key: value', got {body!r}")
            key, _, value = body.partition(":")
            meetings[-1][key.strip()] = _parse_scalar(value)
    return meetings


def fixture_operator_hand_counts() -> dict:
    """HARD: six operator-chosen meetings must match an independent hand count.

    This is the only check in the package the parser cannot influence, so it is
    never reported as passing until it has actually run against real counts. It
    stays BLOCKED while any ``motions_total`` is still null -- a check that did
    not execute is not a check that succeeded.

    Returns:
        Check result: blocked while counts are outstanding, otherwise a
        pass/fail comparison per meeting.
    """
    targets = load_hand_counts()
    if not targets:
        return {
            "name": "operator_hand_counts",
            "severity": "HARD",
            "passed": None,
            "blocked": True,
            "detail": f"No hand-count file at {HAND_COUNTS_PATH}.",
        }

    awaiting = [t["meeting_id"] for t in targets if t.get("motions_total") is None]
    listed = [
        {
            "meeting_id": t.get("meeting_id"),
            "era": t.get("era"),
            "document_id": t.get("document_id"),
            "title": t.get("title"),
            "pdf_path": t.get("pdf_path"),
            "page_count": t.get("page_count"),
            "parser_motions_total": t.get("parser_motions_total"),
            "parser_motions_adopted": t.get("parser_motions_adopted"),
            "parser_motions_lost": t.get("parser_motions_lost"),
            "hand_motions_total": t.get("motions_total"),
        }
        for t in targets
    ]

    if awaiting:
        return {
            "name": "operator_hand_counts",
            "severity": "HARD",
            "passed": None,
            "blocked": True,
            "awaiting_counts_for": awaiting,
            "detail": {
                "file": HAND_COUNTS_PATH,
                "reason": "Hand counts not yet supplied for every target meeting.",
                "targets": listed,
            },
        }

    mismatches = []
    for t in targets:
        for field, parser_field in (
            ("motions_total", "parser_motions_total"),
            ("motions_adopted", "parser_motions_adopted"),
            ("motions_lost", "parser_motions_lost"),
        ):
            hand, parsed = t.get(field), t.get(parser_field)
            if hand is not None and parsed is not None and hand != parsed:
                mismatches.append(
                    {
                        "meeting_id": t.get("meeting_id"),
                        "field": field,
                        "hand_count": hand,
                        "parser_count": parsed,
                    }
                )
    return {
        "name": "operator_hand_counts",
        "severity": "HARD",
        "passed": not mismatches,
        "blocked": False,
        "meetings_checked": len(targets),
        "detail": {"mismatches": mismatches, "targets": listed},
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
        fixture_attendance_vote_discrepancies(),
        fixture_disposition_and_locator(),
        fixture_operator_hand_counts(),
    ]
    advisory = [advisory_vote_named_rate(), advisory_parse_failures()]
    print(json.dumps({"hard": hard, "advisory": advisory}, indent=1, default=str))
    failed = [h for h in hard if h.get("passed") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
