"""Phase 0 reconnaissance for the minutes fact tables.

Read-only. Pulls minutes text from Postgres via ``podman exec psql`` (no
psycopg2 dependency yet -- the pip install is pending operator approval) and
reports inventory, attachment offset, vote formats, format eras and roster
availability.

No LLM is involved in any date, name, vote, motion or count path: every value
below comes from regex and arithmetic over the stored document text.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date

CONTAINER = "boarddocs-postgres"
DB_USER = "boarddocs"
DB_NAME = "boarddocs"

# Record separator that will not occur in PDF-extracted minutes text.
ROW_SEP = "\x1e"
COL_SEP = "\x1f"


def psql(sql: str) -> list[list[str]]:
    """Run a read-only query and return rows as lists of string columns.

    Args:
        sql: SQL text to execute.

    Returns:
        One list of column strings per result row.
    """
    cmd = [
        "podman",
        "exec",
        "-i",
        CONTAINER,
        "psql",
        "-U",
        DB_USER,
        "-d",
        DB_NAME,
        "-t",
        "-A",
        "-R",
        ROW_SEP,
        "-F",
        COL_SEP,
        "-c",
        sql,
    ]
    # S603: the argument vector is entirely literal apart from ``sql``, there
    # is no shell, and every caller in this package passes a static read-only
    # query defined in module source -- no external input reaches this call.
    # This shell-out is a Phase 0 stopgap; it is replaced by psycopg2 once the
    # dependency install is approved.
    out = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for raw in out.split(ROW_SEP):
        raw = raw.strip("\n")
        if raw.strip():
            rows.append(raw.split(COL_SEP))
    return rows


# --- date extraction from filenames -------------------------------------
# Minutes filenames encode the date of the meeting the minutes describe,
# which is usually NOT the meeting the file is attached to.
FNAME_PATTERNS = [
    # Board Minutes 2025 02 26.pdf  /  Board Minutes 2025-02-26.pdf
    (
        re.compile(r"(20\d{2})[ _\-+]?(0[1-9]|1[0-2])[ _\-+]?(0[1-9]|[12]\d|3[01])\b"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
    ),
    # BoardMeetingMinutes102208.pdf -> MMDDYY
    (
        re.compile(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{2})(?!\d)"),
        lambda m: (2000 + int(m.group(3)), int(m.group(1)), int(m.group(2))),
    ),
]


def date_from_filename(title: str) -> date | None:
    """Extract the minutes' own meeting date from the attachment filename.

    Args:
        title: Attachment filename, e.g. ``Board_Minutes_042413.pdf``.

    Returns:
        Parsed date, or None when no pattern matches or the date is invalid.
    """
    stem = re.sub(r"\.pdf$", "", title, flags=re.I)
    for rx, conv in FNAME_PATTERNS:
        m = rx.search(stem)
        if m:
            try:
                y, mo, d = conv(m)
                if 2004 <= y <= 2027:
                    return date(y, mo, d)
            except ValueError:
                continue
    return None


# --- date extraction from the minutes body -------------------------------
MONTHS = ("January February March April May June July August September " "October November December").split()
MONTH_RX = "|".join(MONTHS)
BODY_DATE_RXS = [
    re.compile(rf"\b({MONTH_RX})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b"),
    re.compile(rf"\b(\d{{1,2}})\s+({MONTH_RX})\s+(20\d{{2}})\b"),
]


def date_from_body(text: str, window: int = 1500) -> date | None:
    """Extract the meeting date stated near the top of the minutes body.

    Args:
        text: Full extracted text of the minutes document.
        window: Number of leading characters to search.

    Returns:
        The first parseable date in the header window, else None.
    """
    head = text[:window]
    for rx in BODY_DATE_RXS:
        m = rx.search(head)
        if not m:
            continue
        try:
            if m.re is BODY_DATE_RXS[0]:
                mo = MONTHS.index(m.group(1)) + 1
                return date(int(m.group(3)), mo, int(m.group(2)))
            mo = MONTHS.index(m.group(2)) + 1
            return date(int(m.group(3)), mo, int(m.group(1)))
        except ValueError:
            continue
    return None


# --- format era detection ------------------------------------------------
ERA_SIGNALS = {
    "motion_no": re.compile(r"Motion\s+No\.\s*\d+\s*-\s*\d+", re.I),
    "passive_motion": re.compile(r"A motion was made", re.I),
    "motion_carried": re.compile(r"Motion\s+carried", re.I),
    "the_motion_carried": re.compile(r"The motion carried", re.I),
    "rollcall_attend": re.compile(r"Roll Call", re.I),
    "moved_by": re.compile(r"(moved by|motion by)", re.I),
    "yea_named": re.compile(
        r"\bYea:",
    ),
    "present_colon": re.compile(r"(President|Director|Vice President)\s+\w+:\s*Present"),
}


def era_signature(text: str) -> tuple[str, ...]:
    """Compute the set of layout signals present in a minutes document.

    Args:
        text: Full extracted text of the minutes document.

    Returns:
        Sorted tuple of signal names that fired.
    """
    return tuple(sorted(k for k, rx in ERA_SIGNALS.items() if rx.search(text)))


# --- executive session ---------------------------------------------------
EXEC_RX = re.compile(r"[^.\n]*executive session[^.]*\.", re.I)
RCW_RX = re.compile(r"RCW\s*42\.30\.110\s*\(?1\)?\s*\(?([a-z])\)?", re.I)


def main() -> None:
    """Run all Phase 0 checks and print a JSON blob of the findings."""
    findings: dict[str, object] = {}

    rows = psql("""
        SELECT id, title, to_char(meeting_date,'YYYY-MM-DD'),
               coalesce(meeting_id,''), length(content_text), content_text
        FROM documents
        WHERE document_type='attachment' AND title ILIKE '%minutes%'
        ORDER BY meeting_date
    """)

    docs = []
    for r in rows:
        if len(r) < 6:
            continue
        docs.append(
            {
                "id": r[0],
                "title": r[1],
                "attached_to": r[2],
                "meeting_id": r[3],
                "len": int(r[4]) if r[4].isdigit() else 0,
                "text": COL_SEP.join(r[5:]),
            }
        )
    findings["minutes_attachment_count"] = len(docs)

    # --- item 3: attachment offset --------------------------------------
    offsets = Counter()
    offset_examples = []
    body_vs_fname_mismatch = []
    for d in docs:
        bdate = date_from_body(d["text"])
        fdate = date_from_filename(d["title"])
        d["body_date"], d["fname_date"] = bdate, fdate
        if bdate and fdate and bdate != fdate:
            body_vs_fname_mismatch.append((d["title"], str(bdate), str(fdate)))
        eff = bdate or fdate
        d["effective_date"] = eff
        if eff and d["attached_to"]:
            try:
                att = date.fromisoformat(d["attached_to"])
            except ValueError:
                continue
            delta = (att - eff).days
            offsets[delta] += 1
            if len(offset_examples) < 25:
                offset_examples.append(
                    {
                        "title": d["title"],
                        "minutes_date": str(eff),
                        "attached_to": d["attached_to"],
                        "offset_days": delta,
                    }
                )
    findings["offset_histogram"] = dict(sorted(offsets.items(), key=lambda kv: -kv[1])[:20])
    findings["offset_examples"] = offset_examples
    findings["body_vs_filename_mismatch_count"] = len(body_vs_fname_mismatch)
    findings["body_vs_filename_mismatch_sample"] = body_vs_fname_mismatch[:15]
    findings["body_date_parsed"] = sum(1 for d in docs if d["body_date"])
    findings["fname_date_parsed"] = sum(1 for d in docs if d["fname_date"])

    # --- item 5: format eras --------------------------------------------
    sig_years = defaultdict(list)
    for d in docs:
        eff = d["effective_date"]
        if not eff:
            continue
        sig_years[era_signature(d["text"])].append((eff, d["title"], d["id"]))
    eras = []
    for sig, items in sorted(sig_years.items(), key=lambda kv: -len(kv[1])):
        items.sort()
        eras.append(
            {
                "signals": list(sig),
                "n": len(items),
                "first": str(items[0][0]),
                "last": str(items[-1][0]),
                "sample_title": items[len(items) // 2][1],
                "sample_doc_id": items[len(items) // 2][2],
            }
        )
    findings["era_signatures"] = eras[:15]

    # --- item 4: vote format per document -------------------------------
    # A real tally must sit next to vote language. Bare "N-N" is rejected:
    # in this corpus it is always a motion number ("Motion No. 03-10"), a
    # school year ("2005-06") or a grade range ("1-12").
    tally_rx = re.compile(
        r"(?:vote|votes|voting|tally|carried|passed|failed)\D{0,20}\b(\d)\s*[-to/]{1,3}\s*(\d)\b", re.I
    )
    motion_no_rx = re.compile(r"Motion\s+No\.\s*\d+\s*-\s*\d+", re.I)
    passive_rx = re.compile(r"A motion was made", re.I)
    dispo_rx = re.compile(r"motion\s+(carried|passed|failed|lost)", re.I)

    vote_fmt_by_year = defaultdict(Counter)
    motions_by_year = Counter()
    docs_with_motions = Counter()
    docs_by_year = Counter()
    for d in docs:
        eff = d["effective_date"]
        if not eff:
            continue
        t = d["text"]
        docs_by_year[eff.year] += 1
        n_motions = len(motion_no_rx.findall(t)) or len(passive_rx.findall(t))
        motions_by_year[eff.year] += n_motions
        if n_motions:
            docs_with_motions[eff.year] += 1

        if re.search(r"\bYea:", t):
            fmt = "named"
        elif re.search(r"(Director|President|Vice President)\s+\w+:\s*(Yes|No|Aye|Nay)\b", t, re.I):
            fmt = "roll_call"
        elif tally_rx.search(t):
            fmt = "tally_only"
        elif dispo_rx.search(t):
            fmt = "carried_no_names"
        elif n_motions:
            fmt = "motion_no_disposition"
        else:
            fmt = "no_motion_language"
        vote_fmt_by_year[eff.year][fmt] += 1
    findings["vote_format_by_year"] = {y: dict(c) for y, c in sorted(vote_fmt_by_year.items())}
    findings["motions_by_year"] = dict(sorted(motions_by_year.items()))
    findings["docs_with_motions_by_year"] = dict(sorted(docs_with_motions.items()))
    findings["minutes_docs_by_body_year"] = dict(sorted(docs_by_year.items()))

    # --- executive sessions by minutes-body year ------------------------
    exec_by_year = Counter()
    exec_docs_by_year = Counter()
    rcw_cited = Counter()
    exec_samples = []
    for d in docs:
        eff = d["effective_date"]
        if not eff:
            continue
        hits = EXEC_RX.findall(d["text"])
        if hits:
            exec_docs_by_year[eff.year] += 1
            exec_by_year[eff.year] += len(hits)
            if eff.year == 2024 and len(exec_samples) < 40:
                for h in hits:
                    exec_samples.append(
                        {"date": str(eff), "title": d["title"], "sentence": re.sub(r"\s+", " ", h).strip()[:300]}
                    )
        for m in RCW_RX.finditer(d["text"]):
            rcw_cited[m.group(1).lower()] += 1
    findings["exec_session_announcements_by_year"] = dict(sorted(exec_by_year.items()))
    findings["exec_session_docs_by_year"] = dict(sorted(exec_docs_by_year.items()))
    findings["exec_2024_samples"] = exec_samples
    findings["rcw_subsection_citations"] = dict(rcw_cited)

    print(json.dumps(findings, indent=1, default=str))


if __name__ == "__main__":
    main()
