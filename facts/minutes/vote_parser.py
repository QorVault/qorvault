"""Parser for named votes recorded in BoardDocs agenda items.

Why this module exists
----------------------
Phase 0 established that the minutes PDFs do not record how individual
directors voted. Across all 874 minutes documents spanning 2005-2026, exactly
two record named votes and two more use a roll-call format; every other motion
reads "Motion carried." with no mover, no second, no names and no tally.

The named votes live in the BoardDocs ``agenda_item`` documents instead, in a
``Motion & Voting`` block covering 2018-2026::

    Motion & Voting
    A motion was made to approve Resolution No. 1697 - District Budget Adoption.
    Motion by Donald Cook, second by Tim Clark.
    Final Resolution: Motion Carries
    Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song, Teresa Gregory
    A motion was made to amend the proposed budget ...
    Motion by Donald Cook, second by Andy Song.
    Final Resolution: Motion Fails
    Yea: Donald Cook
    Nay: Tim Clark, Meghin Margel, Andy Song
    Abstain: Teresa Gregory

A single agenda item may carry several motions, so ``Final Resolution:`` is
used as the anchor for each one rather than assuming one motion per document.

Rows produced here carry ``source='agenda_item'`` and a locator pointing at the
agenda-item document, keeping the locator rule intact: document_id, page and a
verbatim quote.

No LLM is involved in any name, vote or count path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from locators import make_quote

MOTION_VOTING_RX = re.compile(r"Motion\s*&\s*Voting|Motion\s+and\s+Voting", re.I)
FINAL_RES_RX = re.compile(r"Final\s+Resolution\s*:\s*Motion\s+(Carries|Fails)", re.I)
MADE_RX = re.compile(r"A\s+motion\s+was\s+made\b", re.I)
MOVER_RX = re.compile(
    r"Motion\s+by\s+([^,\n]{2,60}?)\s*,\s*second(?:ed)?\s+by\s+([^.\n]{2,60}?)\s*\.",
    re.I,
)
MOVER_ONLY_RX = re.compile(r"Motion\s+by\s+([^.,\n]{2,60}?)\s*\.", re.I)
ROLL_RX = re.compile(r"^\s*(Yea|Nay|Abstain|Absent)\s*:\s*([^\n]+)$", re.M)
CONSENT_RX = re.compile(r"consent\s+(agenda|calendar)", re.I)

DISPOSITION_MAP = {"carries": "adopted", "fails": "lost"}
VOTE_MAP = {"yea": "yes", "nay": "no", "abstain": "abstain", "absent": "absent"}


@dataclass
class AgendaVote:
    """One director's recorded vote on one motion."""

    director_raw: str
    vote: str
    offset: int
    quote: str


@dataclass
class AgendaMotion:
    """A motion recorded in an agenda item's Motion & Voting block."""

    seq: int
    motion_text: str
    mover_raw: str | None
    second_raw: str | None
    disposition: str
    vote_format: str
    is_consent_agenda: bool
    offset: int
    quote: str
    tally_yes: int | None
    tally_no: int | None
    tally_abstain: int | None
    votes: list[AgendaVote] = field(default_factory=list)


def _split_names(blob: str) -> list[str]:
    """Split a comma-separated roll into individual director names.

    Args:
        blob: Text after ``Yea:``/``Nay:``/``Abstain:``.

    Returns:
        Trimmed names, empty entries removed.
    """
    out = []
    for part in blob.split(","):
        name = re.sub(r"\s+", " ", part).strip(" .;:")
        if not name or len(name) > 60:
            continue
        # When a roll is printed without line breaks surviving extraction, a
        # later label can be swept into the name ("Nay: Song"). A director name
        # never contains a vote label or a colon.
        if ":" in name:
            continue
        if re.match(r"^(Yea|Nay|Abstain|Absent)\b", name, re.I):
            continue
        if not re.match(r"^[A-Z]", name):
            continue
        out.append(name)
    return out


def parse_agenda_item(text: str) -> list[AgendaMotion]:
    """Extract motions and named votes from one agenda-item document.

    Each ``Final Resolution:`` line anchors one motion. The motion text is the
    nearest preceding "A motion was made" sentence; mover and second are read
    from the ``Motion by ..., second by ...`` line between them; the vote rolls
    are the ``Yea:``/``Nay:``/``Abstain:``/``Absent:`` lines that follow.

    Args:
        text: Full extracted text of the agenda-item document.

    Returns:
        Motions in document order. Empty when the item was never voted on --
        informational items have no Motion & Voting block, which is valid.
    """
    if not MOTION_VOTING_RX.search(text):
        return []

    anchors = list(FINAL_RES_RX.finditer(text))
    if not anchors:
        return []

    made = [m.start() for m in MADE_RX.finditer(text)]
    out: list[AgendaMotion] = []

    for i, anchor in enumerate(anchors):
        # Motion text: nearest "A motion was made" before this resolution that
        # is after the previous resolution.
        prev_end = anchors[i - 1].end() if i else 0
        candidates = [p for p in made if prev_end <= p < anchor.start()]
        start = candidates[-1] if candidates else prev_end
        head = text[start : anchor.start()]

        mv = MOVER_RX.search(head)
        if mv:
            mover, second = mv.group(1).strip(), mv.group(2).strip()
        else:
            mo = MOVER_ONLY_RX.search(head)
            mover, second = (mo.group(1).strip(), None) if mo else (None, None)

        # Motion text is the head with the mover line and any embedded roll
        # call removed, so motion_text stays the verbatim proposition.
        body = MOVER_RX.sub("", head)
        body = MOVER_ONLY_RX.sub("", body)
        body = re.sub(
            r"^\s*(?:President|Vice\s+President|Director)\s+[^\n:]{1,40}:\s*" r"(?:Yea|Nay|Abstain|Absent)\s*$",
            "",
            body,
            flags=re.M | re.I,
        )
        body = re.sub(r"Voting took place via roll call vote\.?", "", body, flags=re.I)
        motion_text = re.sub(r"\s+", " ", body).strip()
        if not motion_text:
            motion_text = make_quote(text, start, anchor.start())

        # Vote rolls: the canonical full-name rolls sit immediately after the
        # Final Resolution line. Stop at the NEXT motion's opening, not at the
        # next Final Resolution -- some blocks also print a surname-only
        # "ROLL CALL VOTE" roll BEFORE their resolution::
        #
        #     A motion was made to ...
        #     ROLL CALL VOTE
        #     Yea: Gregory, Clark, Margel      <- motion i+1's pre-roll
        #     Motion by Tim Clark, second by Andy Song.
        #     Final Resolution: Motion Carries
        #     Yea: Tim Clark, Meghin Margel    <- motion i+1's canonical roll
        #
        # Running to the next resolution would attribute motion i+1's pre-roll
        # to motion i and double its tally.
        next_anchor = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        tail_start = anchor.end()
        following = [p for p in made if p > tail_start]
        next_start = min(following[0], next_anchor) if following else next_anchor
        tail = text[tail_start:next_start]

        votes: list[AgendaVote] = []
        counts = {"yes": 0, "no": 0, "abstain": 0}
        seen: set[str] = set()
        for rm in ROLL_RX.finditer(tail):
            vote = VOTE_MAP[rm.group(1).lower()]
            for name in _split_names(rm.group(2)):
                if name in seen:
                    continue
                seen.add(name)
                votes.append(
                    AgendaVote(
                        director_raw=name,
                        vote=vote,
                        offset=tail_start + rm.start(),
                        quote=make_quote(text, tail_start + rm.start(), tail_start + rm.end()),
                    )
                )
                if vote in counts:
                    counts[vote] += 1

        disposition = DISPOSITION_MAP[anchor.group(1).lower()]
        out.append(
            AgendaMotion(
                seq=len(out) + 1,
                motion_text=motion_text[:4000],
                mover_raw=mover,
                second_raw=second,
                disposition=disposition,
                vote_format="named" if votes else "carried_no_names",
                is_consent_agenda=bool(CONSENT_RX.search(motion_text[:400])),
                offset=start,
                # Quote spans the motion through its disposition so the locator
                # resolves to a page containing the disposition word.
                quote=make_quote(text, start, anchor.end(), max_len=400),
                tally_yes=counts["yes"] if votes else None,
                tally_no=counts["no"] if votes else None,
                tally_abstain=counts["abstain"] if votes else None,
                votes=votes,
            )
        )
    return out
