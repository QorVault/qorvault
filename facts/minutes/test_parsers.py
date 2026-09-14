"""Unit tests for the minutes parsers.

These run against fixed text samples, not the database, so they pin parser
behaviour independently of corpus state. Every sample is real text shape taken
from the corpus during Phase 0.
"""

from __future__ import annotations

from datetime import date

import pytest
from census import classify
from dates import date_from_body, date_from_filename
from parsers import (
    detect_era,
    detect_meeting_type,
    parse_attendance,
    parse_exec_sessions,
    parse_motions,
)
from vote_parser import parse_agenda_item

ERA_A_MINUTES = """KENT SCHOOL DISTRICT NO. 415
KING COUNTY
KENT, WASHINGTON

June 8, 2011

The Board of Directors of Kent School District No. 415 met at 4:15 p.m.,
Wednesday, June 8, 2011, in the boardroom of the Administration Center located
at 12033 SE 256 th Street, Kent, Washington, with President Bill Boyce
presiding. Other board members present: Jim Berrios, Tim Clark, Karen DeBruler
and Debbie Straus. Also present: Superintendent Dr. Edward Lee Vargas.

Recess

President Boyce recessed the meeting at 6:00 p.m. for an executive session for
approximately 30 minutes to discuss personnel evaluations.

Motion No. 51-11 That the Board of Directors approves adoption of Revised
Policy 3207: Prohibition of Harassment, Intimidation, and Bullying.

  Motion carried.
"""

ERA_B_MINUTES = """KENT SCHOOL DISTRICT NO. 415
KING COUNTY
KENT, WASHINGTON

Board Meeting Minutes
February 26, 2025

Call to Order
President Meghin Margel called the regular meeting to order at 6:34 p.m.
Roll Call
President Margel: Present
Vice President Cook: Present
Director Clark: Present
Director Song: Present (attended virtually)
Director Gregory: Absent/Excused
Also present: Board Secretary, Superintendent Vela

Agenda Review
A motion was made to approve the agenda as presented.
The motion carried.
"""

AGENDA_ITEM = """Motion & Voting
A motion was made to approve Resolution No. 1697 - District Budget Adoption.
Motion by Donald Cook, second by Tim Clark.
Final Resolution: Motion Carries
Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song, Teresa Gregory
A motion was made to amend the proposed budget.
ROLL CALL VOTE
Yea: Gregory, Clark, Margel
Nay: Song, Cook
Motion by Donald Cook, second by Andy Song.
Final Resolution: Motion Fails
Yea: Donald Cook
Nay: Tim Clark, Meghin Margel, Andy Song
Abstain: Teresa Gregory
"""


class TestDates:
    """Date extraction from filenames and document bodies."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Board Minutes 2025 02 26.pdf", date(2025, 2, 26)),
            ("BoardMeetingMinutes102208.pdf", date(2008, 10, 22)),
            ("Board_Minutes_042413.pdf", date(2013, 4, 24)),
            ("Board+Meeting+Minutes+060910.pdf", date(2010, 6, 9)),
        ],
    )
    def test_filename_dates(self, title, expected):
        """Filenames encode the minutes' own meeting date."""
        assert date_from_filename(title) == expected

    def test_body_date_wins_over_header_noise(self):
        """The body header date is parsed from the top of the document."""
        assert date_from_body(ERA_A_MINUTES) == date(2011, 6, 8)
        assert date_from_body(ERA_B_MINUTES) == date(2025, 2, 26)

    def test_unparseable_returns_none(self):
        """No date means None, never a guess."""
        assert date_from_filename("Vouchers.pdf") is None
        assert date_from_body("no date here at all") is None


class TestEra:
    """Era selection by meeting date."""

    def test_era_boundary(self):
        """Era A ends 2022-05-11; Era B begins 2022-08-24."""
        assert detect_era(date(2022, 5, 11)) == "A"
        assert detect_era(date(2022, 8, 24)) == "B"
        assert detect_era(date(2005, 1, 1)) == "A"
        assert detect_era(date(2026, 1, 1)) == "B"


class TestAttendance:
    """Attendance extraction in both eras."""

    def test_era_a_prose_attendance(self):
        """Presiding officer and the members-present list are both captured."""
        att = parse_attendance(ERA_A_MINUTES, "A")
        names = {a.director_raw for a in att}
        assert "Bill Boyce" in names
        assert {"Jim Berrios", "Tim Clark", "Karen DeBruler", "Debbie Straus"} <= names
        assert len(att) == 5

    def test_era_a_captures_presiding_with_board_title(self):
        """'Board President X presiding' must not drop the president."""
        text = ERA_A_MINUTES.replace("with President Bill Boyce", "with Board President Bill Boyce")
        assert any(a.director_raw == "Bill Boyce" for a in parse_attendance(text, "A"))

    def test_era_a_strips_parenthetical_qualifier(self):
        """'(via telephone)' sets status and is stripped from the name."""
        text = ERA_A_MINUTES.replace("Debbie Straus.", "Debbie Straus (via telephone).")
        att = {a.director_raw: a.status for a in parse_attendance(text, "A")}
        assert att.get("Debbie Straus") == "present_virtual"

    def test_era_b_roll_call(self):
        """The roll call block yields one row per director with status."""
        att = {a.director_raw: a.status for a in parse_attendance(ERA_B_MINUTES, "B")}
        assert att["Margel"] == "present"
        assert att["Song"] == "present_virtual"
        assert att["Gregory"] == "excused"
        assert len(att) == 5

    def test_era_b_roll_call_is_not_a_vote(self):
        """Era B 'Roll Call' is attendance; it must not produce motions."""
        assert parse_motions(ERA_B_MINUTES, "B")[0].vote_format == "carried_no_names"


class TestMotions:
    """Motion extraction and disposition mapping."""

    def test_era_a_numbered_motion(self):
        """Numbered motions carry their number and a disposition."""
        motions = parse_motions(ERA_A_MINUTES, "A")
        assert len(motions) == 1
        assert motions[0].motion_number_raw == "Motion No. 51-11"
        assert motions[0].disposition == "adopted"
        assert "Policy 3207" in motions[0].motion_text

    def test_locator_quote_contains_disposition(self):
        """The locator must evidence the outcome, not just the motion."""
        quote = parse_motions(ERA_A_MINUTES, "A")[0].quote
        assert "carried" in quote.lower()

    def test_long_motion_quote_still_reaches_disposition(self):
        """A long consent motion must not truncate before its disposition."""
        filler = "item " * 400
        text = ERA_A_MINUTES.replace("Policy 3207", "Policy 3207 " + filler)
        quote = parse_motions(text, "A")[0].quote
        assert "carried" in quote.lower()
        assert "…" in quote

    def test_era_b_passive_motion(self):
        """Passive motions are detected without a motion number."""
        motions = parse_motions(ERA_B_MINUTES, "B")
        assert len(motions) == 1
        assert motions[0].motion_number_raw is None
        assert motions[0].disposition == "adopted"

    def test_no_motions_is_valid(self):
        """A work session with no motions parses to an empty list."""
        assert parse_motions("A work session was held. No action taken.", "B") == []


class TestExecSessions:
    """Executive session announcements."""

    def test_recess_to_exec_session(self):
        """A recess announcement produces one row with its purpose."""
        sessions = parse_exec_sessions(ERA_A_MINUTES)
        assert len(sessions) == 1
        assert "personnel evaluations" in sessions[0].announced_purpose

    def test_consent_agenda_line_is_not_an_announcement(self):
        """Approving PRIOR minutes is not an executive session announcement."""
        text = "9.13 - Minutes of 29 November 2023 Executive Session was " "approved."
        assert parse_exec_sessions(text) == []

    def test_closing_sets_end_time_not_a_new_session(self):
        """An adjournment is an end time, not an additional session."""
        text = (
            "President Margel announced an Executive Session for "
            "approximately three minutes at 10:09 p.m. "
            "The Executive Session was adjourned at 10:25 p.m."
        )
        sessions = parse_exec_sessions(text)
        assert len(sessions) == 1
        assert sessions[0].actual_end_time == "10:25 p.m."

    def test_purpose_category_only_when_rcw_cited(self):
        """purpose_category is never inferred from prose."""
        assert parse_exec_sessions(ERA_A_MINUTES)[0].purpose_category is None
        text = (
            "President Boyce recessed the meeting for an executive "
            "session to discuss litigation pursuant to "
            "RCW 42.30.110(1)(i) for 30 minutes."
        )
        assert parse_exec_sessions(text)[0].purpose_category == "42.30.110(1)(i)"


class TestAgendaVotes:
    """Named votes from BoardDocs agenda items."""

    def test_two_motions_parsed(self):
        """Each Final Resolution anchors its own motion."""
        motions = parse_agenda_item(AGENDA_ITEM)
        assert len(motions) == 2
        assert motions[0].disposition == "adopted"
        assert motions[1].disposition == "lost"

    def test_mover_and_second(self):
        """Mover and second are read from the motion line."""
        motions = parse_agenda_item(AGENDA_ITEM)
        assert motions[0].mover_raw == "Donald Cook"
        assert motions[0].second_raw == "Tim Clark"

    def test_named_votes_not_double_counted(self):
        """A preceding ROLL CALL VOTE roll must not inflate the prior motion.

        The first motion has five yea votes. Running the vote scan to the next
        Final Resolution would sweep in the second motion's surname-only
        pre-roll and report ten.
        """
        first = parse_agenda_item(AGENDA_ITEM)[0]
        assert first.tally_yes == 5
        assert len(first.votes) == 5
        assert all(" " in v.director_raw for v in first.votes)

    def test_second_motion_tally(self):
        """Yea/Nay/Abstain are counted separately."""
        second = parse_agenda_item(AGENDA_ITEM)[1]
        assert (second.tally_yes, second.tally_no, second.tally_abstain) == (1, 3, 1)

    def test_vote_labels_never_become_names(self):
        """A label swept into a name is rejected rather than stored."""
        text = AGENDA_ITEM.replace("Nay: Tim Clark", "Nay: Tim Clark, Nay: Song")
        for motion in parse_agenda_item(text):
            for v in motion.votes:
                assert ":" not in v.director_raw

    def test_no_voting_block(self):
        """An informational item yields no motions."""
        assert parse_agenda_item("AGENDA ITEM: informational only.") == []


class TestMeetingType:
    """Meeting type detection and census classification."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Board Special Meeting Minutes 2024 01 10 work session.pdf", "work_study"),
            ("Board Minutes 2024 01 10.pdf", "regular"),
            ("Board Minutes 2024 07 10 Special Meeting.pdf", "special"),
        ],
    )
    def test_detect_meeting_type(self, title, expected):
        """Filename drives the meeting type."""
        assert detect_meeting_type(title, "") == expected

    @pytest.mark.parametrize(
        "slug,expected",
        [
            ("regular-meeting-6-30-p-m-", "regular"),
            ("special-meeting-executive-session-5-30-p-m-", "exec_session"),
            ("special-meeting-work-session-4-30-p-m-", "work_study"),
            ("notice-this-meeting-has-been-canceled-due-to-weather", "non_meeting"),
            ("board-members-attending-mlk-rallies", "non_meeting"),
        ],
    )
    def test_census_classify(self, slug, expected):
        """Cancellations and ceremonial appearances are not meetings."""
        assert classify(slug) == expected
