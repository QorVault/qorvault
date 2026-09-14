"""Meeting census from the scraped BoardDocs meeting directories.

Why a census is needed
----------------------
``facts.meeting`` cannot be built from minutes alone: a meeting whose minutes
are missing would simply not exist, and the required ``meetings_missing_minutes``
view would have nothing to report. The census supplies the denominator.

The ``agenda`` document type is not usable for this -- it holds only 3 rows for
2019 and 4 for 2020. The scraped meeting directory names are complete for
2005-2026 and encode both the meeting date and the meeting type::

    2024-01-10-special-meeting-executive-session-5-30-p-m-6-30-p-m-
    2024-01-24-regular-meeting-6-30-p-m-

Cancellation notices are excluded: a meeting that was cancelled did not happen
and owes no minutes.
"""

from __future__ import annotations

import os
import re
from datetime import date

MEETING_ROOT = "/home/donald/qorvault-dev-archive/framework-backup/home/" "ksd_forensic/boarddocs/data"

DIR_RX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.*)$")

# Slugs that are not board meetings owing minutes: cancellations, ceremonial
# appearances, conference attendance, community events.
NON_MEETING = re.compile(
    r"(canceled|cancelled|cancelation|cancellation|reception|wssda|"
    r"community-connect|chat-with-the|board-members-attending|retirement|"
    r"graduation|open-house|ribbon|groundbreaking|tour\b)",
    re.I,
)


def classify(slug: str) -> str:
    """Classify a meeting directory slug into a meeting type.

    Order matters: an executive session scheduled as a "special meeting -
    executive session" is an executive session, and a "special meeting - work
    session" is a work session.

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


def load_census(root: str = MEETING_ROOT) -> list[dict]:
    """Read the meeting census from the scraped directory names.

    A single date can host several distinct meetings (a work session, the
    regular meeting, and an executive session all on one evening), and can even
    host two executive sessions at different times. Each directory is one
    meeting occurrence.

    Args:
        root: Directory holding the scraped meeting folders.

    Returns:
        One dict per real meeting, with ``date``, ``type`` and ``slug``.
    """
    out: list[dict] = []
    if not os.path.isdir(root):
        return out
    # The same meeting is occasionally scraped twice under differently
    # punctuated slugs ("...-work-session-5-00-p-m-" and
    # "...-work-session-500-pm"). Those are one meeting, not two. Comparing
    # slugs with punctuation stripped collapses the re-scrapes while keeping
    # genuinely distinct same-day meetings (an executive session at 4:30 and
    # another at 7:00) apart, because their times differ.
    seen: set[str] = set()
    for name in sorted(os.listdir(root)):
        m = DIR_RX.match(name)
        if not m:
            continue
        mtype = classify(m.group(4))
        if mtype == "non_meeting":
            continue
        try:
            mdate = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        fingerprint = re.sub(r"[^a-z0-9]", "", name.lower())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append({"date": mdate, "type": mtype, "slug": name})
    return out
