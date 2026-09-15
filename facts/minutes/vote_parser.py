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
# Vote roll labels. "Aye" is an affirmative synonym for "Yea": it appears in
# exactly one agenda item in the corpus (1479b410, the 2025-12-10 board
# reorganization), where every officer election is recorded with Aye/Nay. Not
# recognising it meant those rolls were read as Nay-only, which is how a
# unanimous 5-0 vote for Vice President came through as a lone "None" voting no.
ROLL_RX = re.compile(r"^\s*(Aye|Yea|Nay|Abstain|Absent)\s*:\s*([^\n]+)$", re.M)
CONSENT_RX = re.compile(r"consent\s+(agenda|calendar)", re.I)
# "Nay: None." records that nobody voted that way. It is a count of zero, not a
# director. Matched as the ENTIRE roll value so a real name is never dropped.
NONE_ROLL_RX = re.compile(r"^\s*none\s*\.?\s*$", re.I)
# A later block heading ends the current motion's block. "Recommended Action"
# opens the next agenda item's narrative; "Motion & Voting" opens the next
# voting block.
RECOMMENDED_ACTION_RX = re.compile(r"Recommended\s+Action", re.I)
# A line that is part of a vote roll: the label lines themselves, or a blank
# line separating them. Anything else ends the roll.
ROLL_LINE_RX = re.compile(r"^\s*(?:Aye|Yea|Nay|Abstain|Absent)\s*:", re.I)

# Locator quote ceiling for an agenda-item motion.
#
# The quote must bridge the motion opening to its "Final Resolution:" line so
# the citation resolves to text that actually shows the outcome. Measured over
# all 4,214 agenda-item motions in the corpus, that span is 122 chars at
# minimum, 152 at the median and 3,518 at the maximum. The previous 400-char
# ceiling truncated 45 of them before the disposition ever appeared, which is
# what made `disposition_and_locator` fail on 43 rows. 4,000 clears the longest
# span in the corpus with headroom and matches the existing motion_text cap.
MOTION_QUOTE_MAX_LEN = 4000

DISPOSITION_MAP = {"carries": "adopted", "fails": "lost"}
VOTE_MAP = {"aye": "yes", "yea": "yes", "nay": "no", "abstain": "abstain", "absent": "absent"}


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
        blob: Text after ``Aye:``/``Yea:``/``Nay:``/``Abstain:``.

    Returns:
        Trimmed names, empty entries removed. Empty when the roll records
        nobody.
    """
    # "Nay: None." means nobody voted that way -- a count of zero. Read as a
    # name it produced a director called "None" voting against seating the
    # Vice President. Checked against the whole roll value, so a director whose
    # name merely contained the word would be unaffected.
    if NONE_ROLL_RX.match(blob):
        return []
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
        if re.match(r"^(Aye|Yea|Nay|Abstain|Absent)\b", name, re.I):
            continue
        # A bare "None" inside a comma-separated roll is also a count of zero.
        if NONE_ROLL_RX.match(name):
            continue
        if not re.match(r"^[A-Z]", name):
            continue
        out.append(name)
    return out


def _canonical_roll_block(tail: str) -> str:
    """Trim a motion's tail to the contiguous vote roll that belongs to it.

    A motion's canonical roll is the run of ``Yea:``/``Nay:``/``Abstain:``/
    ``Absent:`` lines immediately following its ``Final Resolution:`` line. The
    first line that is neither a roll line nor blank ends it.

    This is what stops a roll from bleeding across motions. The 2025-02-11
    special meeting is the case that forced it: one ``Final Resolution:``
    anchor, followed by prose and then eight ``ROLL CALL VOTING ROUND`` blocks
    of *nomination* votes. Those rounds are not votes on the motion, but with
    no later anchor and no later "A motion was made" to stop at, the scan ran
    to the end of the document and swept them in -- recording 8 votes from a
    4-member board.

    Args:
        tail: Document text starting immediately after the ``Final
            Resolution:`` line and already bounded by the next block.

    Returns:
        The leading portion of ``tail`` holding only this motion's roll.
    """
    out: list[str] = []
    seen_roll = False
    for line in tail.splitlines(keepends=True):
        if ROLL_LINE_RX.match(line):
            seen_roll = True
            out.append(line)
            continue
        if not line.strip():
            # Blank lines are permitted inside a roll and before it, but never
            # carry content.
            out.append(line)
            continue
        # Any other content ends the roll -- and, before the roll has started,
        # means this motion has none below its resolution. No prose is tolerated
        # in between: measured over the corpus, 4,210 of 4,214 motions put the
        # roll on the first non-blank line after the resolution, and not one
        # needs a gap. The only motions with a roll further down are the
        # 2025-12-10 elections, where that roll belongs to the NEXT motion --
        # so reaching across prose would reintroduce the off-by-one this
        # function exists to stop.
        break
    return "".join(out) if seen_roll else ""


def _preceding_roll_block(text: str, lower_bound: int, anchor_start: int) -> tuple[int, str]:
    """Find the contiguous vote roll that sits just BEFORE a resolution line.

    Some items print the roll above its ``Final Resolution:`` line instead of
    below it. The 2025-12-10 board reorganization is the corpus's only example:
    each officer election reads as narrative, then ``Aye:``/``Nay:`` lines, then
    the resolution. Read with the usual below-the-line assumption, every motion
    received the *following* motion's roll -- a systematic off-by-one.

    Used only as a fallback, when a motion has no roll after its resolution.
    Where a roll appears both above and below (the surname-only pre-roll shape
    that belongs to the next motion), the one below wins and this is never
    consulted.

    Args:
        text: Full document text.
        lower_bound: Offset the scan may not cross. This is the end of the
            previous motion's consumed roll, which stops a motion with no roll
            of its own from claiming the previous motion's.
        anchor_start: Start offset of this motion's ``Final Resolution:`` line.

    Returns:
        ``(absolute_offset, block_text)`` for the roll, or ``(anchor_start,
        "")`` when there is none.
    """
    head = text[lower_bound:anchor_start]
    lines = head.splitlines(keepends=True)
    kept: list[str] = []
    seen_roll = False
    # Walk backwards from the resolution: the roll is whatever sits immediately
    # above it. The first non-blank, non-roll line above ends it.
    for line in reversed(lines):
        if ROLL_LINE_RX.match(line):
            seen_roll = True
            kept.append(line)
            continue
        if not line.strip():
            kept.append(line)
            continue
        break
    if not seen_roll:
        return anchor_start, ""
    block = "".join(reversed(kept))
    return lower_bound + len(head) - len(block), block


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
    # End offset of the roll already consumed by the previous motion. The
    # look-back for an above-the-line roll may not cross it.
    prev_roll_end = 0

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
        bounds = [next_anchor]
        if following:
            bounds.append(following[0])
        # A later block heading also ends this motion's block. Without this the
        # bound falls back to end-of-document whenever a motion is the last one
        # in its item.
        for heading_rx in (MOTION_VOTING_RX, RECOMMENDED_ACTION_RX):
            heading = heading_rx.search(text, tail_start)
            if heading:
                bounds.append(heading.start())
        next_start = min(bounds)
        # Bounding by the next block is necessary but not sufficient: a
        # nomination roll-call sequence can sit inside this same block. Keep
        # only the contiguous roll that follows the resolution.
        tail = _canonical_roll_block(text[tail_start:next_start])
        roll_start = tail_start
        if not tail:
            # No roll below the resolution. Some items print it above instead,
            # so look there before concluding the motion has no named vote.
            # `prev_roll_end` bounds the look-back so a motion that genuinely
            # has no roll cannot claim the previous motion's.
            roll_start, tail = _preceding_roll_block(text, prev_roll_end, anchor.start())

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
                        offset=roll_start + rm.start(),
                        quote=make_quote(text, roll_start + rm.start(), roll_start + rm.end()),
                    )
                )
                if vote in counts:
                    counts[vote] += 1

        prev_roll_end = max(prev_roll_end, anchor.end(), roll_start + len(tail))

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
                # resolves to a page containing the disposition word. The cap
                # clears the longest such span in the corpus -- a quote that
                # stops short of the resolution cites the right document but
                # proves nothing about the outcome.
                quote=make_quote(text, start, anchor.end(), max_len=MOTION_QUOTE_MAX_LEN),
                tally_yes=counts["yes"] if votes else None,
                tally_no=counts["no"] if votes else None,
                tally_abstain=counts["abstain"] if votes else None,
                votes=votes,
            )
        )
    return out
