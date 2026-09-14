"""Deterministic parsers for Kent School District board minutes.

One parser per format era, selected by the meeting date stated in the minutes
body. No LLM touches any date, name, vote, motion or count in this module:
everything below is regex and arithmetic over the extracted text.

Format eras established in Phase 0
----------------------------------
Era A -- "numbered motion", 2004-08-31 through 2022-05-11.
    Motions carry a year-scoped sequence number and a standalone disposition
    line::

        Motion No. 51-11 That the Board of Directors approves adoption of
        Revised Policy 3207: Prohibition of Harassment, Intimidation, and
        Bullying.

        Motion carried.

    Attendance is prose in the opening paragraph ("...with President Bill
    Boyce presiding. Other board members present: Jim Berrios, Tim Clark...").

Era B -- "roll call / passive motion", 2022-08-24 onward.
    A ``Roll Call`` heading introduces a structured attendance block, and
    motions are passive and unnumbered::

        A motion was made to approve the agenda as presented.
        The motion carried.

The last Era A document is dated 2022-05-11 and the first Era B document
2022-08-24; no document uses both conventions, so a cut at 2022-07-01 routes
every document correctly.

Neither era records how individual directors voted. Vote rows come from
BoardDocs agenda items instead -- see ``vote_parser.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from locators import make_quote

ERA_CUTOVER = date(2022, 7, 1)

# --------------------------------------------------------------- meetings --
MEETING_TYPE_RX = [
    (re.compile(r"\bregular\s+(?:board\s+)?meeting\b", re.I), "regular"),
    (re.compile(r"\bwork\s+session\b|\bstudy\s+session\b", re.I), "work_study"),
    (re.compile(r"\bexecutive\s+session\b", re.I), "exec_session"),
    (re.compile(r"\bspecial\s+meeting\b", re.I), "special"),
]


def detect_era(meeting_date: date) -> str:
    """Return the format era for a meeting date.

    Args:
        meeting_date: Date stated in the minutes body.

    Returns:
        ``"A"`` or ``"B"``.
    """
    return "A" if meeting_date < ERA_CUTOVER else "B"


def detect_meeting_type(title: str, text: str) -> str:
    """Classify the meeting type from the filename and the minutes header.

    The filename is checked first because it is unambiguous ("Board Special
    Meeting Minutes ... work session.pdf"); the document header is the
    fallback.

    Args:
        title: Attachment filename.
        text: Full extracted text of the minutes.

    Returns:
        One of ``regular``, ``special``, ``work_study``, ``exec_session``,
        ``other``.
    """
    blob = f"{title}\n{text[:800]}"
    # Order matters: a "Special Meeting ... work session" is a work session.
    if re.search(r"work\s*session|study\s*session", blob, re.I):
        return "work_study"
    if re.search(r"executive\s*session", blob, re.I):
        return "exec_session"
    if re.search(r"special\s*meeting", blob, re.I):
        return "special"
    if re.search(r"regular\s*meeting|board\s*minutes|board\s*meeting", blob, re.I):
        return "regular"
    return "other"


# ------------------------------------------------------------- attendance --
# Era B: "Director Cook: Absent/Excused", "Director Song: Present (attended
# virtually)", "President X: Absent at the time of roll call; arrived at ~5:03"
ERA_B_ATTEND_RX = re.compile(
    r"^\s*(President|Vice\s+President|Director)\s+"
    r"([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)\s*:\s*"
    r"([^\n]{0,120})$",
    re.M,
)

# Era A prose: "with President Bill Boyce presiding. Other board members
# present: Jim Berrios, Tim Clark, Karen DeBruler and Debbie Straus."
# The presiding officer's title varies: "President", "Vice President",
# "Board President", "Board Vice President". Missing the optional "Board"
# silently drops the presiding director from attendance.
ERA_A_PRESIDING_RX = re.compile(
    r"with\s+(?:Board\s+)?(?:Vice\s+)?President\s+"
    r"([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)*)\s+"
    r"(?:presiding|calling\s+the\s+meeting)",
    re.I,
)

# "Michele Bettinger attended via teleconference." -- a director listed in a
# sentence of their own rather than in the "members present" list.
ERA_A_ATTENDED_VIA_RX = re.compile(
    r"\b([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})\s+attended\s+"
    r"(?:the\s+meeting\s+)?(?:via|by|through)\s+"
    r"(teleconference|telephone|phone|zoom|video|virtually)",
    re.I,
)
ERA_A_PRESENT_RX = re.compile(
    r"(?:Other\s+board\s+members\s+present|Board\s+members\s+present|" r"Members\s+present)\s*:?\s*([^.]{0,300})\.",
    re.I,
)
ERA_A_ABSENT_RX = re.compile(r"(?:Board\s+members?\s+absent|Absent)\s*:?\s*([^.]{0,200})\.", re.I)

# "Denise Daniels was excused." / "Director Song was absent."
ERA_A_EXCUSED_RX = re.compile(
    r"\b([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})\s+was\s+" r"(excused|absent)\b",
)

# Names carry attendance qualifiers in parentheses: "Debbie Straus (via
# telephone)". Keep the qualifier out of the name and let it set the status.
PAREN_RX = re.compile(r"\s*\(([^)]*)\)\s*$")
VIRTUAL_RX = re.compile(r"telephone|phone|virtual|remote|zoom", re.I)


def _clean_name(raw: str) -> tuple[str, str | None]:
    """Split an attendance qualifier off a director's name.

    Args:
        raw: Name as printed, e.g. ``"Debbie Straus (via telephone)"``.

    Returns:
        Tuple of (clean name, status override or None).
    """
    name = raw.strip(" .;:")
    status = None
    m = PAREN_RX.search(name)
    if m:
        qualifier = m.group(1)
        name = PAREN_RX.sub("", name).strip()
        if VIRTUAL_RX.search(qualifier):
            status = "present_virtual"
        elif re.search(r"excus", qualifier, re.I):
            status = "excused"
        elif re.search(r"absent", qualifier, re.I):
            status = "absent"
    return name, status


def _split_names(blob: str) -> list[tuple[str, str | None]]:
    """Split a prose name list into individual names.

    Args:
        blob: Text such as ``"Jim Berrios, Tim Clark and Debbie Straus"``.

    Returns:
        List of (name, status override) pairs, empty entries removed.
    """
    blob = re.sub(r"\s+", " ", blob).strip()
    parts = re.split(r",| and | & ", blob)
    out = []
    for p in parts:
        name, status = _clean_name(p)
        # Reject obvious non-names: titles carried along, or sentence tails.
        if not name or len(name) > 60:
            continue
        if not re.match(r"^[A-Z]", name):
            continue
        out.append((name, status))
    return out


def _era_b_status(raw: str) -> str | None:
    """Map an Era B roll-call value onto an attendance status.

    Args:
        raw: Text after the colon, e.g. ``"Absent/Excused"``.

    Returns:
        A status accepted by ``facts.attendance.status``, or None if the line
        is not an attendance value (e.g. "Vacant at this time").
    """
    low = raw.lower()
    if "vacant" in low:
        return None
    if "arrived" in low or "joined the meeting" in low:
        return "arrived_late"
    if "excused" in low:
        return "excused"
    if low.startswith("absent"):
        return "absent"
    if "virtual" in low or "remotely" in low:
        return "present_virtual"
    if low.startswith("present"):
        return "present"
    return None


@dataclass
class Attendance:
    """One director's attendance at one meeting."""

    director_raw: str
    role_raw: str | None
    status: str
    offset: int
    quote: str


def parse_attendance(text: str, era: str) -> list[Attendance]:
    """Extract per-director attendance from a minutes document.

    Args:
        text: Full extracted text of the minutes.
        era: ``"A"`` or ``"B"``.

    Returns:
        Attendance records; empty when the document states no attendance.
    """
    out: list[Attendance] = []
    seen: set[str] = set()

    if era == "B":
        for m in ERA_B_ATTEND_RX.finditer(text):
            role, name, value = m.group(1), m.group(2), m.group(3).strip()
            status = _era_b_status(value)
            if status is None:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(
                Attendance(
                    director_raw=name,
                    role_raw=role,
                    status=status,
                    offset=m.start(),
                    quote=make_quote(text, m.start(), m.end()),
                )
            )
        if out:
            return out
        # Some Era B documents print the block without line breaks surviving
        # extraction; fall through to the prose parser rather than returning
        # nothing.

    head = text[:2500]
    m = ERA_A_PRESIDING_RX.search(head)
    if m:
        name = m.group(1).strip()
        if name not in seen:
            seen.add(name)
            out.append(
                Attendance(
                    director_raw=name,
                    role_raw="President",
                    status="present",
                    offset=m.start(),
                    quote=make_quote(text, m.start(), m.end()),
                )
            )
    m = ERA_A_PRESENT_RX.search(head)
    if m:
        for name, override in _split_names(m.group(1)):
            if name in seen:
                continue
            seen.add(name)
            out.append(
                Attendance(
                    director_raw=name,
                    role_raw=None,
                    status=override or "present",
                    offset=m.start(),
                    quote=make_quote(text, m.start(), m.end()),
                )
            )
    m = ERA_A_ABSENT_RX.search(head)
    if m:
        for name, _ in _split_names(m.group(1)):
            if name in seen:
                continue
            seen.add(name)
            out.append(
                Attendance(
                    director_raw=name,
                    role_raw=None,
                    status="absent",
                    offset=m.start(),
                    quote=make_quote(text, m.start(), m.end()),
                )
            )
    for m in ERA_A_ATTENDED_VIA_RX.finditer(head):
        name = m.group(1).strip()
        if name in seen:
            continue
        seen.add(name)
        out.append(
            Attendance(
                director_raw=name,
                role_raw=None,
                status="present_virtual",
                offset=m.start(),
                quote=make_quote(text, m.start(), m.end()),
            )
        )
    # "Denise Daniels was excused." sits outside the present/absent lists.
    for m in ERA_A_EXCUSED_RX.finditer(head):
        name = m.group(1).strip()
        if name in seen:
            continue
        seen.add(name)
        out.append(
            Attendance(
                director_raw=name,
                role_raw=None,
                status="excused" if m.group(2).lower() == "excused" else "absent",
                offset=m.start(),
                quote=make_quote(text, m.start(), m.end()),
            )
        )
    return out


# ----------------------------------------------------------------- motions --
MOTION_NO_RX = re.compile(r"Motion\s+No\.\s*(\d+\s*-\s*\d+)", re.I)
PASSIVE_MOTION_RX = re.compile(r"A\s+motion\s+was\s+made\b", re.I)
DISPOSITION_RX = re.compile(
    r"\b(?:the\s+)?motion\s+(carried|passed|failed|lost|was\s+withdrawn|" r"was\s+tabled|died)\b",
    re.I,
)
TABLED_RX = re.compile(r"\b(?:motion\s+to\s+)?table\b", re.I)
WITHDRAWN_RX = re.compile(r"\bwithdrew\b|\bwithdrawn\b", re.I)
CONSENT_RX = re.compile(r"consent\s+(agenda|calendar)", re.I)
AMENDED_RX = re.compile(r"\bas\s+amended\b", re.I)

DISPOSITION_MAP = {
    "carried": "adopted",
    "passed": "adopted",
    "failed": "lost",
    "lost": "lost",
    "died": "lost",
    "was withdrawn": "withdrawn",
    "was tabled": "tabled",
}


@dataclass
class Motion:
    """A single motion recorded in the minutes."""

    seq: int
    motion_number_raw: str | None
    motion_text: str
    pre_amendment_text: str | None
    mover_raw: str | None
    second_raw: str | None
    disposition: str
    vote_format: str
    is_consent_agenda: bool
    offset: int
    quote: str
    tally_yes: int | None = None
    tally_no: int | None = None
    tally_abstain: int | None = None


def _normalize_disposition(word: str) -> str:
    """Map a disposition verb onto the schema's disposition enum.

    Args:
        word: Captured disposition word, e.g. ``"carried"``.

    Returns:
        One of ``adopted``, ``lost``, ``tabled``, ``withdrawn``.
    """
    key = re.sub(r"\s+", " ", word.lower().strip())
    return DISPOSITION_MAP.get(key, "adopted")


def parse_motions(text: str, era: str) -> list[Motion]:
    """Extract motions and their dispositions from a minutes document.

    Motion text runs from the motion marker to its disposition line; multi-line
    motion bodies are joined. A consent agenda adopting many items is recorded
    as ONE motion whose text carries the item list, not one motion per item.

    Args:
        text: Full extracted text of the minutes.
        era: ``"A"`` or ``"B"``.

    Returns:
        Motions in document order. Empty is valid -- work sessions and
        executive sessions routinely contain no motions.
    """
    starts: list[tuple[int, str | None]] = []
    if era == "A":
        for m in MOTION_NO_RX.finditer(text):
            starts.append((m.start(), m.group(1).replace(" ", "")))
    if not starts:
        for m in PASSIVE_MOTION_RX.finditer(text):
            starts.append((m.start(), None))
    if not starts:
        return []

    starts.sort()
    out: list[Motion] = []
    for i, (start, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        block = text[start:end]

        dm = DISPOSITION_RX.search(block)
        if dm:
            disposition = _normalize_disposition(dm.group(1))
            body_end = start + dm.start()
            dispo_end = start + dm.end()
        else:
            # No disposition in this block. Withdrawn/tabled motions sometimes
            # say so without the word "motion"; otherwise skip -- a motion with
            # no disposition is not a fact we can assert.
            if WITHDRAWN_RX.search(block):
                disposition, body_end, dispo_end = "withdrawn", end, end
            elif TABLED_RX.search(block[:200]):
                disposition, body_end, dispo_end = "tabled", end, end
            else:
                continue

        body = text[start:body_end]
        motion_text = re.sub(r"\s+", " ", body).strip()
        # Trim the leading marker so motion_text starts at the substance.
        motion_text = re.sub(r"^Motion\s+No\.\s*\d+\s*-\s*\d+\s*", "", motion_text, flags=re.I)
        if not motion_text:
            continue

        out.append(
            Motion(
                seq=len(out) + 1,
                motion_number_raw=f"Motion No. {number}" if number else None,
                motion_text=motion_text[:4000],
                pre_amendment_text=None,
                mover_raw=None,
                second_raw=None,
                disposition=disposition,
                vote_format="carried_no_names",
                is_consent_agenda=bool(CONSENT_RX.search(motion_text[:400])),
                offset=start,
                # The quote must CONTAIN the disposition word: a citation that
                # resolves only to the motion's opening does not evidence the
                # outcome. Consent-agenda motions run to thousands of characters,
                # so a fixed-length excerpt from the start would truncate long
                # before "Motion carried." Bridge the opening to the disposition
                # sentence instead.
                quote=_bridge_quote(text, start, body_end, dispo_end),
            )
        )
    return out


def _bridge_quote(
    text: str, start: int, body_end: int, dispo_end: int, head_len: int = 160, tail_len: int = 120
) -> str:
    """Build a quote spanning a motion's opening and its disposition.

    Args:
        text: Full document text.
        start: Offset where the motion begins.
        body_end: Offset where the motion body ends (disposition starts).
        dispo_end: Offset where the disposition phrase ends.
        head_len: Characters to take from the motion opening.
        tail_len: Characters of disposition context to take.

    Returns:
        A quote containing both the motion opening and the disposition word,
        elided with an ellipsis when the body was longer than ``head_len``.
    """
    head = re.sub(r"\s+", " ", text[start : min(body_end, start + head_len)]).strip()
    tail = re.sub(r"\s+", " ", text[max(body_end, dispo_end - tail_len) : dispo_end]).strip()
    if body_end <= start + head_len:
        return f"{head} {tail}".strip()
    return f"{head} … {tail}".strip()


# ------------------------------------------------------- executive sessions --
# Genuine announcements. Deliberately narrow: the phrase "executive session"
# also appears in consent-agenda line items approving PRIOR minutes
# ("08 - Minutes of 13 December 2023 Regular Meeting, and Executive Session"),
# which are not announcements and must not produce rows.
# The gap between the verb and "executive session" must tolerate periods:
# "recessed the meeting at 6:00 p.m. for an executive session" contains one
# inside "p.m.". Excluding periods here silently dropped every announcement
# that stated a start time. Newlines still bound the match.
# A sentence ends at a period followed by whitespace or end-of-text. A bare
# "[^.]" terminator breaks on the periods inside "p.m." and "RCW 42.30.110(1)(i)",
# truncating the very values the row needs.
# Minutes wrap lines mid-sentence, so the gaps must tolerate newlines as well
# as the periods inside "p.m." and "RCW 42.30.110(1)(i)". A sentence ends at a
# period followed by whitespace or end-of-text; a bare "[^.]" terminator broke
# on both abbreviations and truncated the values these rows need. Lengths stay
# bounded so a match cannot run away across paragraphs.
EXEC_ANNOUNCE_RX = re.compile(
    r"[^.\n]{0,120}?"
    r"\b(?:recessed|adjourned|announced|convened|reconvened|extended|extension)\b"
    r"[\s\S]{0,160}?\bexecutive\s+session\b[\s\S]{0,220}?\.(?=\s|$)",
    re.I,
)

# The closing sentence puts the verb AFTER the noun ("The Executive Session was
# adjourned at 10:25 p.m."), so EXEC_ANNOUNCE_RX does not match it. It is an end
# time for the session already open, NOT a new session: counting it as its own
# announcement would inflate the executive-session count by one per session.
EXEC_CLOSE_RX = re.compile(
    r"[^.\n]{0,80}?\bexecutive\s+session\b[\s\S]{0,60}?"
    r"\b(?:was\s+)?(?:adjourned|ended|concluded|closed)\b[\s\S]{0,80}?\.(?=\s|$)",
    re.I,
)
EXEC_CONSENT_NOISE_RX = re.compile(r"minutes\s+of\b|^\s*\d{1,2}\s*[.\-–]", re.I)

# Sentence matching stops at the first period, which falls INSIDE "p.m.", so
# the captured text routinely ends "at 10:09 p." -- the trailing "m." is on the
# far side of the boundary. Accept both forms and normalize on the way out.
EXEC_TIME_RX = re.compile(r"\bat\s+(\d{1,2}:\d{2}\s*[ap]\.?\s*m?\.?)", re.I)


def _normalize_time(raw: str) -> str:
    """Normalize a clock time captured from minutes text.

    Args:
        raw: Time as matched, e.g. ``"10:09 p."`` or ``"10:25 p.m."``.

    Returns:
        Normalized time such as ``"10:09 p.m."``.
    """
    t = re.sub(r"\s+", " ", raw).strip().rstrip(".")
    m = re.match(r"^(\d{1,2}:\d{2})\s*([ap])\.?\s*m?$", t, re.I)
    if m:
        return f"{m.group(1)} {m.group(2).lower()}.m."
    return t


EXEC_DURATION_RX = re.compile(r"for\s+(?:approximately\s+)?([\w\-]+(?:\s+\w+)?)\s*(minutes?|hours?|hour)", re.I)
EXEC_PURPOSE_RX = re.compile(r"\bto\s+(discuss|rule\s+on|consider|review)\b([^.]{0,200})", re.I)
RCW_RX = re.compile(r"RCW\s*42\.30\.110\s*\(\s*1\s*\)\s*\(\s*([a-z])\s*\)", re.I)


@dataclass
class ExecSession:
    """One executive-session announcement entered in the minutes."""

    seq: int
    announced_purpose: str | None
    purpose_category: str | None
    announced_at: str | None
    stated_end_time: str | None
    actual_end_time: str | None
    is_extension: bool
    announcement_kind: str
    offset: int
    quote: str


def parse_exec_sessions(text: str) -> list[ExecSession]:
    """Extract executive-session announcements from a minutes document.

    One row per announcement: an extension or a reconvening is a separate
    announcement and therefore a separate row.

    ``purpose_category`` is populated only when the text explicitly cites an
    RCW 42.30.110(1) subsection. Phase 0 found exactly one such citation in the
    whole corpus, so this column is almost always NULL by design -- the purpose
    is not inferred from the prose.

    Args:
        text: Full extracted text of the minutes.

    Returns:
        Executive-session announcements in document order.
    """
    out: list[ExecSession] = []
    for m in EXEC_ANNOUNCE_RX.finditer(text):
        sentence = re.sub(r"\s+", " ", m.group(0)).strip()
        if EXEC_CONSENT_NOISE_RX.search(sentence):
            continue

        low = sentence.lower()
        if "exten" in low:
            kind, is_ext = "extension", True
        elif "reconvened" in low:
            kind, is_ext = "reconvene", False
        elif "adjourned" in low:
            kind, is_ext = "adjourn", False
        else:
            kind, is_ext = "in_meeting", False

        purpose = None
        pm = EXEC_PURPOSE_RX.search(sentence)
        if pm:
            purpose = re.sub(r"\s+", " ", pm.group(0)).strip(" .")

        rcw = RCW_RX.search(sentence)
        times = EXEC_TIME_RX.findall(sentence)

        out.append(
            ExecSession(
                seq=len(out) + 1,
                announced_purpose=purpose,
                purpose_category=f"42.30.110(1)({rcw.group(1).lower()})" if rcw else None,
                announced_at=_normalize_time(times[0]) if times and kind != "adjourn" else None,
                stated_end_time=None,
                actual_end_time=_normalize_time(times[0]) if times and kind == "adjourn" else None,
                is_extension=is_ext,
                announcement_kind=kind,
                offset=m.start(),
                quote=sentence[:300],
            )
        )

    # Attach closing times to the session they close rather than emitting a row.
    for m in EXEC_CLOSE_RX.finditer(text):
        sentence = re.sub(r"\s+", " ", m.group(0)).strip()
        if EXEC_CONSENT_NOISE_RX.search(sentence):
            continue
        tm = EXEC_TIME_RX.search(sentence)
        if not tm:
            continue
        # The session being closed is the last one announced before this point.
        prior = [s for s in out if s.offset < m.start()]
        if prior and prior[-1].actual_end_time is None:
            prior[-1].actual_end_time = _normalize_time(tm.group(1))
    return out


@dataclass
class ParsedMinutes:
    """Everything extracted from one minutes document."""

    meeting_type: str
    era: str
    attendance: list[Attendance] = field(default_factory=list)
    motions: list[Motion] = field(default_factory=list)
    exec_sessions: list[ExecSession] = field(default_factory=list)


def parse_document(title: str, text: str, meeting_date: date) -> ParsedMinutes:
    """Parse one minutes document end to end.

    Args:
        title: Attachment filename.
        text: Full extracted text of the minutes.
        meeting_date: Date stated in the minutes body.

    Returns:
        The parsed contents of the document.
    """
    era = detect_era(meeting_date)
    return ParsedMinutes(
        meeting_type=detect_meeting_type(title, text),
        era=era,
        attendance=parse_attendance(text, era),
        motions=parse_motions(text, era),
        exec_sessions=parse_exec_sessions(text),
    )
