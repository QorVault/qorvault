"""Build the minutes fact tables and load them into the ``facts`` schema.

Reads (read-only, enforced by the database session) from ``documents``, parses
with the deterministic era parsers, and loads via ``COPY ... FROM STDIN`` so no
document text is ever interpolated into SQL.

Sources
-------
``documents`` where ``document_type='attachment'`` and title matches ``%minutes%``
    Meetings, attendance, motions and executive sessions.

``documents`` where ``document_type='agenda_item'`` with a ``Motion & Voting``
block
    Named votes, movers and seconds, 2018-2026 only. The minutes do not record
    these -- see ``vote_parser`` for the evidence.

Meeting identity is ``<ISO date>:<meeting type>``, shared by both sources so
agenda-item votes attach to the meeting whose minutes describe it.

Nothing is ever deleted from the corpus. ``--reload`` truncates only the
``facts`` tables this package owns.

Usage::

    set -a && source ~/workspace/projects/ksd-main/.env && set +a
    export PGPASSWORD=...        # .env's Postgres password is stale
    .venv/bin/python build.py --reload
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from datetime import date

import db
from census import load_census
from dates import date_from_body, date_from_filename
from locators import build_page_map
from parsers import detect_era, parse_document
from vote_parser import parse_agenda_item

FACT_TABLES = [
    "facts.vote",
    "facts.motion",
    "facts.attendance",
    "facts.executive_session",
    "facts.meeting",
    "facts.minutes_parse_log",
]

MOTION_COLS = [
    "motion_id",
    "meeting_id",
    "motion_seq",
    "motion_number_raw",
    "agenda_item_ref",
    "agenda_item_order",
    "motion_text",
    "pre_amendment_text",
    "mover_raw",
    "second_raw",
    "disposition",
    "tally_yes",
    "tally_no",
    "tally_abstain",
    "vote_format",
    "is_consent_agenda",
    "source",
    "locator_document_id",
    "locator_page",
    "locator_char_offset",
    "locator_quote",
]


def copy_into(conn, table: str, columns: list[str], rows: list[list]) -> int:
    """Load rows into a table with ``COPY ... FROM STDIN``.

    Values travel as CSV on the copy stream rather than being interpolated into
    SQL text, so verbatim minutes content cannot affect the statement.

    Args:
        conn: Open psycopg2 connection.
        table: Fully qualified table name (a module constant, never user input).
        columns: Column names in the order the rows provide them.
        rows: Row values.

    Returns:
        Number of rows submitted.
    """
    if not rows:
        return 0
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    buf.seek(0)
    collist = ", ".join(columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({collist}) FROM STDIN WITH (FORMAT csv, NULL '')",
            buf,
        )
    return len(rows)


def meeting_key(meeting_date: date, meeting_type: str) -> str:
    """Build the stable meeting identifier.

    Args:
        meeting_date: Date of the meeting itself.
        meeting_type: Coarse meeting type.

    Returns:
        Identifier of the form ``2025-02-26:regular``.
    """
    return f"{meeting_date.isoformat()}:{meeting_type}"


def committee_to_type(committee: str) -> str:
    """Map a BoardDocs committee name onto a meeting type.

    Args:
        committee: ``documents.committee_name`` value.

    Returns:
        A meeting type accepted by ``facts.meeting``.
    """
    low = (committee or "").lower()
    if "work session" in low or "study session" in low:
        return "work_study"
    if "executive session" in low:
        return "exec_session"
    if "regular" in low:
        return "regular"
    if "special" in low:
        return "special"
    return "other"


def load_minutes_documents(limit: int = 0) -> list[dict]:
    """Fetch all minutes attachments with their text and stored path.

    Args:
        limit: Maximum documents to return; 0 means all.

    Returns:
        One dict per document.
    """
    sql = """
        SELECT id::text, title, to_char(meeting_date,'YYYY-MM-DD'),
               coalesce(file_path,''), content_text
        FROM documents
        WHERE document_type='attachment' AND title ILIKE %s
        ORDER BY meeting_date, title
    """
    params: tuple = ("%minutes%",)
    if limit:
        sql += " LIMIT %s"
        params = ("%minutes%", limit)
    return [
        {"id": r[0], "title": r[1], "attached_to": r[2], "file_path": r[3], "text": r[4] or ""}
        for r in db.query(sql, params)
    ]


def load_voted_agenda_items() -> list[dict]:
    """Fetch agenda items that carry a Motion & Voting block.

    Returns:
        One dict per agenda-item document.
    """
    rows = db.query("""
        SELECT id::text, to_char(meeting_date,'YYYY-MM-DD'),
               coalesce(committee_name,''), coalesce(agenda_item_id,''),
               coalesce(file_path,''), content_text
        FROM documents
        WHERE document_type='agenda_item'
          AND content_text ~* 'motion *& *voting'
          AND meeting_date IS NOT NULL
        ORDER BY meeting_date
    """)
    return [
        {
            "id": r[0],
            "meeting_date": r[1],
            "committee": r[2],
            "agenda_item_id": r[3],
            "file_path": r[4],
            "text": r[5] or "",
        }
        for r in rows
    ]


def build_census_meetings() -> dict[str, list]:
    """Seed ``facts.meeting`` from the scraped meeting census.

    Every meeting the district held gets a row whether or not its minutes
    survive, which is what makes ``meetings_missing_minutes`` answerable. Rows
    are later upgraded in place when a minutes document is found for the same
    date and type.

    A single date can host two meetings of the same type (2024-05-29 held
    executive sessions at 4:30 and 7:00 p.m.), so repeat occurrences get a
    ``#2`` suffix rather than colliding on the primary key.

    Returns:
        Meeting rows keyed by meeting id.
    """
    meetings: dict[str, list] = {}
    seen: Counter = Counter()
    for entry in load_census():
        base = meeting_key(entry["date"], entry["type"])
        seen[base] += 1
        key = base if seen[base] == 1 else f"{base}#{seen[base]}"
        meetings[key] = [
            key,
            "kent_sd",
            entry["date"].isoformat(),
            entry["type"],
            None,
            None,
            False,
            None,
            detect_era(entry["date"]),
            None,
            None,
            None,
            None,
            "census",
            entry["slug"],
        ]
    return meetings


def parse_minutes(docs: list[dict]) -> dict:
    """Parse every minutes document into fact rows.

    Args:
        docs: Minutes documents from :func:`load_minutes_documents`.

    Returns:
        Dict of row lists keyed by table, plus per-year statistics.
    """
    meetings: dict[str, list] = build_census_meetings()
    attendance_rows: list[list] = []
    motion_rows: list[list] = []
    exec_rows: list[list] = []
    parse_log: list[list] = []
    per_year: dict[int, Counter] = {}
    stats: Counter = Counter()

    # Pass 1: parse every document and group by the meeting it describes.
    # Several documents can describe one meeting (duplicate scrapes, or a
    # combined PDF re-posted). Child rows must come from exactly one of them,
    # otherwise attendance and motions are double-counted.
    by_key: dict[str, list] = {}
    for doc in docs:
        text, title = doc["text"], doc["title"]

        if not text or len(text) < 500:
            parse_log.append([doc["id"], None, None, "no_text_layer", 0, 0, 0, "content_text under 500 chars"])
            stats["no_text_layer"] += 1
            continue

        mdate = date_from_body(text) or date_from_filename(title)
        if mdate is None:
            parse_log.append([doc["id"], None, None, "date_unresolved", 0, 0, 0, "no date in body header or filename"])
            stats["date_unresolved"] += 1
            continue

        parsed = parse_document(title, text, mdate)
        key = meeting_key(mdate, parsed.meeting_type)
        by_key.setdefault(key, []).append((doc, mdate, parsed))

    # Pass 2: pick the winner per meeting and emit only its rows. The longest
    # text wins: duplicate scrapes of one meeting differ only in completeness.
    for key, candidates in by_key.items():
        candidates.sort(key=lambda c: len(c[0]["text"]), reverse=True)
        winner, losers = candidates[0], candidates[1:]

        for doc, mdate, parsed in losers:
            stats["superseded_duplicate"] += 1
            parse_log.append(
                [
                    doc["id"],
                    mdate.isoformat(),
                    parsed.era,
                    "superseded_duplicate",
                    0,
                    0,
                    0,
                    f"another document covers {key} with more text",
                ]
            )

        doc, mdate, parsed = winner
        text = doc["text"]
        row = per_year.setdefault(mdate.year, Counter())
        row["documents"] += 1

        pmap = build_page_map(text, doc["file_path"])
        row[f"pagemap_{pmap.kind}"] += 1

        # Upgrade the census placeholder in place when one exists, so the
        # census slug is preserved alongside the minutes evidence.
        slug = meetings[key][14] if key in meetings else None
        meetings[key] = [
            key,
            "kent_sd",
            mdate.isoformat(),
            parsed.meeting_type,
            doc["id"],
            doc["attached_to"] or None,
            False,
            None,
            parsed.era,
            doc["id"],
            pmap.page_for_offset(0),
            0,
            text[:200].replace("\n", " ").strip(),
            "minutes",
            slug,
        ]

        for att in parsed.attendance:
            attendance_rows.append(
                [
                    key,
                    att.director_raw,
                    None,
                    att.role_raw,
                    att.status,
                    doc["id"],
                    pmap.page_for_offset(att.offset),
                    att.offset,
                    att.quote,
                ]
            )
        for mo in parsed.motions:
            motion_rows.append(
                [
                    f"{key}#m{mo.seq}",
                    key,
                    mo.seq,
                    mo.motion_number_raw,
                    None,
                    mo.seq,
                    mo.motion_text,
                    mo.pre_amendment_text,
                    mo.mover_raw,
                    mo.second_raw,
                    mo.disposition,
                    mo.tally_yes,
                    mo.tally_no,
                    mo.tally_abstain,
                    mo.vote_format,
                    mo.is_consent_agenda,
                    "minutes",
                    doc["id"],
                    pmap.page_for_offset(mo.offset),
                    mo.offset,
                    mo.quote,
                ]
            )
        for ex in parsed.exec_sessions:
            exec_rows.append(
                [
                    f"{key}#e{ex.seq}",
                    key,
                    ex.seq,
                    ex.announced_purpose,
                    ex.purpose_category,
                    ex.announced_at,
                    ex.stated_end_time,
                    ex.actual_end_time,
                    ex.is_extension,
                    ex.announcement_kind,
                    doc["id"],
                    pmap.page_for_offset(ex.offset),
                    ex.offset,
                    ex.quote,
                ]
            )

        row["motions"] += len(parsed.motions)
        row["exec_sessions"] += len(parsed.exec_sessions)
        row["attendance"] += len(parsed.attendance)
        if parsed.motions:
            row["docs_with_motions"] += 1
        row["parsed"] += 1
        stats["parsed"] += 1
        parse_log.append(
            [
                doc["id"],
                mdate.isoformat(),
                parsed.era,
                "parsed",
                len(parsed.motions),
                len(parsed.exec_sessions),
                len(parsed.attendance),
                f"page_map={pmap.kind}",
            ]
        )

    return {
        "meetings": meetings,
        "attendance": attendance_rows,
        "motions": motion_rows,
        "exec": exec_rows,
        "parse_log": parse_log,
        "per_year": per_year,
        "stats": stats,
    }


def parse_votes(items: list[dict], meetings: dict[str, list]) -> tuple:
    """Parse named votes from agenda items, creating meetings where needed.

    Args:
        items: Agenda-item documents carrying a Motion & Voting block.
        meetings: Meeting rows built from the minutes; extended in place when a
            voted meeting has no minutes in the corpus.

    Returns:
        Tuple of (agenda motion rows, vote rows).
    """
    motion_rows: list[list] = []
    vote_rows: list[list] = []
    seq_by_meeting: Counter = Counter()

    for item in items:
        try:
            idate = date.fromisoformat(item["meeting_date"])
        except (ValueError, TypeError):
            continue
        key = meeting_key(idate, committee_to_type(item["committee"]))
        if key not in meetings:
            # No minutes for this meeting, but the vote is still a fact.
            meetings[key] = [
                key,
                "kent_sd",
                idate.isoformat(),
                committee_to_type(item["committee"]),
                None,
                None,
                False,
                None,
                detect_era(idate),
                item["id"],
                None,
                0,
                f"agenda item {item['agenda_item_id']}",
                "agenda_item",
                None,
            ]
        pmap = build_page_map(item["text"], item["file_path"])

        for am in parse_agenda_item(item["text"]):
            seq_by_meeting[key] += 1
            seq = seq_by_meeting[key]
            motion_id = f"{key}#a{seq}"
            motion_rows.append(
                [
                    motion_id,
                    key,
                    seq,
                    None,
                    item["agenda_item_id"] or None,
                    seq,
                    am.motion_text,
                    None,
                    am.mover_raw,
                    am.second_raw,
                    am.disposition,
                    am.tally_yes,
                    am.tally_no,
                    am.tally_abstain,
                    am.vote_format,
                    am.is_consent_agenda,
                    "agenda_item",
                    item["id"],
                    pmap.page_for_offset(am.offset),
                    am.offset,
                    am.quote,
                ]
            )
            for v in am.votes:
                vote_rows.append(
                    [
                        motion_id,
                        v.director_raw,
                        None,
                        v.vote,
                        "agenda_item",
                        item["id"],
                        pmap.page_for_offset(v.offset),
                        v.offset,
                        v.quote,
                    ]
                )
    return motion_rows, vote_rows


def main() -> int:
    """Parse the corpus and load the fact tables.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description="Build the minutes fact tables.")
    ap.add_argument("--reload", action="store_true", help="truncate the facts tables before loading")
    ap.add_argument("--limit", type=int, default=0, help="parse at most N minutes documents (smoke testing)")
    ap.add_argument("--skip-votes", action="store_true", help="skip the agenda-item vote pass")
    args = ap.parse_args()

    docs = load_minutes_documents(args.limit)
    print(f"minutes documents: {len(docs)}", file=sys.stderr)
    result = parse_minutes(docs)

    if args.skip_votes:
        agenda_motions, votes = [], []
    else:
        items = load_voted_agenda_items()
        print(f"voted agenda items: {len(items)}", file=sys.stderr)
        agenda_motions, votes = parse_votes(items, result["meetings"])

    with db.connect() as conn:
        if args.reload:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE " + ", ".join(FACT_TABLES) + " RESTART IDENTITY CASCADE")
        counts = {
            "meeting": copy_into(
                conn,
                "facts.meeting",
                [
                    "meeting_id",
                    "tenant_id",
                    "meeting_date",
                    "meeting_type",
                    "minutes_document_id",
                    "approved_at_meeting_id",
                    "approved_as_corrected",
                    "correction_note",
                    "format_era",
                    "locator_document_id",
                    "locator_page",
                    "locator_char_offset",
                    "locator_quote",
                    "source",
                    "census_slug",
                ],
                list(result["meetings"].values()),
            ),
            "attendance": copy_into(
                conn,
                "facts.attendance",
                [
                    "meeting_id",
                    "director_raw",
                    "director_norm",
                    "role_raw",
                    "status",
                    "locator_document_id",
                    "locator_page",
                    "locator_char_offset",
                    "locator_quote",
                ],
                result["attendance"],
            ),
            "motion_minutes": copy_into(conn, "facts.motion", MOTION_COLS, result["motions"]),
            "motion_agenda": copy_into(conn, "facts.motion", MOTION_COLS, agenda_motions),
            "vote": copy_into(
                conn,
                "facts.vote",
                [
                    "motion_id",
                    "director_raw",
                    "director_norm",
                    "vote",
                    "source",
                    "locator_document_id",
                    "locator_page",
                    "locator_char_offset",
                    "locator_quote",
                ],
                votes,
            ),
            "executive_session": copy_into(
                conn,
                "facts.executive_session",
                [
                    "exec_session_id",
                    "meeting_id",
                    "announcement_seq",
                    "announced_purpose",
                    "purpose_category",
                    "announced_at",
                    "stated_end_time",
                    "actual_end_time",
                    "is_extension",
                    "announcement_kind",
                    "locator_document_id",
                    "locator_page",
                    "locator_char_offset",
                    "locator_quote",
                ],
                result["exec"],
            ),
            "parse_log": copy_into(
                conn,
                "facts.minutes_parse_log",
                [
                    "document_id",
                    "minutes_date",
                    "format_era",
                    "status",
                    "motions_found",
                    "exec_sessions_found",
                    "attendance_found",
                    "note",
                ],
                result["parse_log"],
            ),
        }

    for name, n in counts.items():
        print(f"loaded {name}: {n}", file=sys.stderr)
    print(f"status counts: {dict(result['stats'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
