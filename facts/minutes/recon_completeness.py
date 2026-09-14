"""Phase 0 item 2: which meetings in the corpus have no minutes document.

The meeting census comes from the scraped BoardDocs meeting directories, not
from the ``agenda`` document type: only 3 agenda rows exist for 2019, so the
agenda table is not a usable denominator. Directory names encode both the
meeting date and the meeting type.

Minutes coverage is the set of meeting dates recovered from minutes
attachments (body date, falling back to the date encoded in the filename).
Read-only; no LLM in any count path.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from dates import date_from_body, date_from_filename
from recon_phase0 import psql

MEETING_ROOT = "/home/donald/qorvault-dev-archive/framework-backup/home/" "ksd_forensic/boarddocs/data"

DIR_RX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.*)$")

# Directory slugs that are not board meetings with minutes obligations.
NON_MEETING = re.compile(
    r"(canceled|cancelled|reception|wssda|community-connect|chat-with-the|"
    r"board-members-attending|retirement|graduation|open-house|ribbon|"
    r"groundbreaking|tour\b)",
    re.I,
)


def classify(slug: str) -> str:
    """Classify a meeting directory slug into a coarse meeting type.

    Args:
        slug: Directory name with the leading ``YYYY-MM-DD-`` stripped.

    Returns:
        One of ``regular``, ``exec_session``, ``work_study``, ``special``,
        or ``non_meeting``.
    """
    s = slug.lower()
    if NON_MEETING.search(s):
        return "non_meeting"
    if "executive-session" in s:
        return "exec_session"
    if "work-session" in s or "study-session" in s:
        return "work_study"
    if "regular-meeting" in s:
        return "regular"
    if "special-meeting" in s:
        return "special"
    return "non_meeting"


def main() -> None:
    """Report meetings lacking minutes, by year and meeting type."""
    census: dict[tuple[str, str], str] = {}
    for name in sorted(os.listdir(MEETING_ROOT)):
        m = DIR_RX.match(name)
        if not m:
            continue
        mdate = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        mtype = classify(m.group(4))
        if mtype == "non_meeting":
            continue
        census[(mdate, mtype)] = name

    minutes_dates: set[str] = set()
    for r in psql("""
        SELECT title, content_text FROM documents
        WHERE document_type='attachment' AND title ILIKE '%minutes%'
    """):
        if len(r) < 2:
            continue
        d = date_from_body("\x1f".join(r[1:])) or date_from_filename(r[0])
        if d:
            minutes_dates.add(d.isoformat())

    by_year: dict[int, Counter] = defaultdict(Counter)
    missing_examples: dict[int, list] = defaultdict(list)
    for (mdate, mtype), dirname in sorted(census.items()):
        year = int(mdate[:4])
        by_year[year][f"{mtype}_total"] += 1
        if mdate in minutes_dates:
            by_year[year][f"{mtype}_with"] += 1
        else:
            by_year[year][f"{mtype}_missing"] += 1
            if mtype == "regular" and len(missing_examples[year]) < 8:
                missing_examples[year].append(dirname)

    out = {
        "meeting_census_total": len(census),
        "minutes_distinct_dates": len(minutes_dates),
        "by_year": {},
        "missing_regular_examples": {k: v for k, v in sorted(missing_examples.items())},
    }
    for y in sorted(by_year):
        c = by_year[y]
        row = {}
        for t in ("regular", "special", "exec_session", "work_study"):
            tot = c.get(f"{t}_total", 0)
            if tot:
                row[t] = {
                    "total": tot,
                    "with_minutes": c.get(f"{t}_with", 0),
                    "missing": c.get(f"{t}_missing", 0),
                    "missing_pct": round(100.0 * c.get(f"{t}_missing", 0) / tot, 1),
                }
        out["by_year"][y] = row

    reg_tot = sum(out["by_year"][y].get("regular", {}).get("total", 0) for y in out["by_year"] if y >= 2015)
    reg_missing = sum(out["by_year"][y].get("regular", {}).get("missing", 0) for y in out["by_year"] if y >= 2015)
    pct = round(100.0 * reg_missing / reg_tot, 1) if reg_tot else None
    out["stop_rule_regular_since_2015"] = {
        "total": reg_tot,
        "missing": reg_missing,
        "missing_pct": pct,
        "threshold_pct": 15.0,
        "breached": bool(pct is not None and pct > 15.0),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
