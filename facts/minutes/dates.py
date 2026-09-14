"""Date extraction from minutes documents.

Two independent sources, deliberately kept separate so they can cross-check
each other:

``date_from_body``
    The meeting date printed in the minutes header. This is authoritative and
    is what ``facts.meeting.meeting_date`` stores.

``date_from_filename``
    The date encoded in the attachment filename (``Board_Minutes_042413.pdf``).
    Used only as a fallback when the body header does not parse.

Neither is ``documents.meeting_date``, which is the date of the LATER meeting
that approved the minutes -- Phase 0 measured 796 of 796 attachment offsets as
positive, dominated by 14 days. Using it as the meeting date would misdate the
whole fact layer by one meeting.

No LLM is involved: these are regexes and calendar arithmetic.
"""

from __future__ import annotations

import re
from datetime import date

FNAME_PATTERNS = [
    # "Board Minutes 2025 02 26.pdf", "Board Minutes 2025-02-26.pdf"
    (
        re.compile(r"(20\d{2})[ _\-+]?(0[1-9]|1[0-2])[ _\-+]?(0[1-9]|[12]\d|3[01])\b"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
    ),
    # "BoardMeetingMinutes102208.pdf" -> MMDDYY
    (
        re.compile(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{2})(?!\d)"),
        lambda m: (2000 + int(m.group(3)), int(m.group(1)), int(m.group(2))),
    ),
]

MONTHS = ("January February March April May June July August September " "October November December").split()
MONTH_RX = "|".join(MONTHS)
BODY_DATE_RXS = [
    re.compile(rf"\b({MONTH_RX})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b"),
    re.compile(rf"\b(\d{{1,2}})\s+({MONTH_RX})\s+(20\d{{2}})\b"),
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
            if rx is BODY_DATE_RXS[0]:
                mo = MONTHS.index(m.group(1)) + 1
                return date(int(m.group(3)), mo, int(m.group(2)))
            mo = MONTHS.index(m.group(2)) + 1
            return date(int(m.group(3)), mo, int(m.group(1)))
        except ValueError:
            continue
    return None
