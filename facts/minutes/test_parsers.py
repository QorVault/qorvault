"""Unit tests for the minutes parsers.

These run against fixed text samples, not the database, so they pin parser
behaviour independently of corpus state. Every sample is real text shape taken
from the corpus during Phase 0.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from census import classify
from dates import date_from_body, date_from_filename
from fixtures import (
    DISPOSITION_WORDS,
    KNOWN_ATTENDANCE_VOTE_DISCREPANCIES,
    _parse_scalar,
    load_hand_counts,
    unknown_attendance_vote_discrepancies,
)
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


class TestDispositionStems:
    """The disposition/quote match is by word stem, not whole word.

    The record inflects the same verb differently in the minutes and in the
    agenda items -- a motion recorded as ``withdrawn`` where the text says the
    mover "withdrew" it. Matching whole words rejected quotes that plainly
    evidenced the outcome.
    """

    @pytest.mark.parametrize(
        "disposition,quote",
        [
            ("adopted", "Motion carried."),
            ("adopted", "Final Resolution: Motion Carries"),
            ("adopted", "The motion will carry on a voice vote."),
            ("adopted", "The motion passed unanimously."),
            ("adopted", "Motion passes 5-0."),
            ("adopted", "on a motion to pass the consent agenda"),
            ("adopted", "the Board adopted Revised Policy 3207"),
            ("adopted", "moved to adopt the 2011-12 budget"),
            ("lost", "The motion failed."),
            ("lost", "Motion fails 2-3."),
            ("lost", "the motion will fail without a second"),
            ("lost", "The motion was lost."),
            ("tabled", "The motion was tabled until June."),
            ("tabled", "moved to table the item"),
            ("withdrawn", "The mover withdrew the motion."),
            ("withdrawn", "The motion was withdrawn."),
            ("withdrawn", "Director Clark moved to withdraw the motion."),
        ],
    )
    def test_stem_matches_inflection(self, disposition, quote):
        """Every inflection of the disposition verb evidences the disposition."""
        assert re.search(DISPOSITION_WORDS[disposition], quote, re.I)

    @pytest.mark.parametrize(
        "disposition,quote",
        [
            # The recommendation put TO the board is not evidence the board
            # adopted it. Admitting "approve" would turn truncated citations
            # green without any of them showing an outcome.
            ("adopted", "Recommended Action That the Board of Directors approves the contracts"),
            ("adopted", "Agenda Item Details Meeting Jun 26, 2019 - Regular Meeting - 7 p.m."),
            # A disposition word for a DIFFERENT outcome must not match.
            ("lost", "Motion carried."),
            ("withdrawn", "Motion carried."),
            ("tabled", "Motion carried."),
        ],
    )
    def test_stem_rejects_non_evidence(self, disposition, quote):
        """Text that does not show the outcome is not accepted as evidence."""
        assert not re.search(DISPOSITION_WORDS[disposition], quote, re.I)


class TestKnownAttendanceVoteDiscrepancies:
    """The discrepancy fixture asserts nothing NEW appears, not that none exist."""

    def test_known_meetings_are_not_reported_as_unknown(self):
        """A discrepancy in a known meeting is accepted."""
        rows = [("2023-12-13:regular", "2023-12-13:regular#a1", 4, 5, "board_transition")]
        assert unknown_attendance_vote_discrepancies(rows) == []

    def test_new_meeting_is_reported(self):
        """A discrepancy outside the known-set is surfaced and fails the check."""
        rows = [("2019-01-09:regular", "2019-01-09:regular#a1", 3, 5, None)]
        unknown = unknown_attendance_vote_discrepancies(rows)
        assert len(unknown) == 1
        assert unknown[0]["meeting_id"] == "2019-01-09:regular"
        assert unknown[0]["cause"] is None

    def test_known_and_unknown_are_separated(self):
        """Known rows are dropped and unknown rows kept, in one pass."""
        rows = [
            ("2022-06-29:special", "2022-06-29:special#a1", 1, 4, "presiding_only"),
            ("2030-01-01:regular", "2030-01-01:regular#a1", 2, 5, None),
            ("2024-07-10:special", "2024-07-10:special#a1", 3, 4, "status_excluded"),
        ]
        assert [u["meeting_id"] for u in unknown_attendance_vote_discrepancies(rows)] == ["2030-01-01:regular"]

    def test_the_fixed_meeting_would_now_be_reported_as_new(self):
        """If the roll bleed regressed, 2025-02-11 would fail the check again."""
        rows = [("2025-02-11:special", "2025-02-11:special#a1", 4, 8, None)]
        assert [u["meeting_id"] for u in unknown_attendance_vote_discrepancies(rows)] == ["2025-02-11:special"]

    def test_known_set_covers_exactly_the_five_diagnosed_meetings(self):
        """The known-set is a closed list; growing it is a deliberate act."""
        assert set(KNOWN_ATTENDANCE_VOTE_DISCREPANCIES) == {
            "2022-06-29:special",
            "2022-10-05:special",
            "2023-11-08:regular",
            "2023-12-13:regular",
            "2024-07-10:special",
        }

    def test_fixed_parser_defect_is_not_still_exempted(self):
        """2025-02-11 was our bug, not the district's; it must not linger.

        Leaving it in the known-set after the parser fix would exempt a
        meeting that no longer needs exempting -- and would silently re-accept
        the defect if it ever regressed.
        """
        assert "2025-02-11:special" not in KNOWN_ATTENDANCE_VOTE_DISCREPANCIES

    def test_every_known_meeting_carries_a_cause(self):
        """A known meeting with no cause would be an unexplained exemption."""
        for meeting_id, (meeting_date, cause) in KNOWN_ATTENDANCE_VOTE_DISCREPANCIES.items():
            assert cause, f"{meeting_id} has no cause"
            assert meeting_id.startswith(meeting_date), f"{meeting_id} date mismatch"

    def test_every_cause_describes_the_record_not_the_parser(self):
        """The known-set exempts record discrepancies, never our own defects."""
        allowed = {"board_transition", "attendance_short", "presiding_only", "status_excluded"}
        for meeting_id, (_date, cause) in KNOWN_ATTENDANCE_VOTE_DISCREPANCIES.items():
            assert cause in allowed, f"{meeting_id}: {cause} is not a record-level cause"


class TestHandCountFile:
    """The hand-count fixture takes the operator's counts from a file."""

    def test_template_lists_six_meetings_three_per_era(self):
        """Phase 0 found two eras, so the split is three and three."""
        targets = load_hand_counts()
        assert len(targets) == 6
        eras = [t["era"] for t in targets]
        assert eras.count("A") == 3
        assert eras.count("B") == 3

    def test_every_target_has_a_resolvable_document_and_pdf(self):
        """A target with no source document cannot be hand counted."""
        for t in load_hand_counts():
            assert t["document_id"]
            assert t["pdf_path"]
            assert isinstance(t["page_count"], int)
            assert isinstance(t["parser_motions_total"], int)

    def test_counts_start_empty_so_the_fixture_stays_blocked(self):
        """The template ships with no counts; the fixture must not go green."""
        for t in load_hand_counts():
            assert t["motions_total"] is None

    def test_scalar_parsing(self):
        """Null becomes None, integers become int, text stays text."""
        assert _parse_scalar("null") is None
        assert _parse_scalar("") is None
        assert _parse_scalar("12") == 12
        assert _parse_scalar("0") == 0
        assert _parse_scalar("A") == "A"
        assert _parse_scalar('"Board Minutes.pdf"') == "Board Minutes.pdf"

    def test_missing_file_yields_no_targets(self, tmp_path):
        """An absent file is blocked, not a crash."""
        assert load_hand_counts(str(tmp_path / "nope.yaml")) == []

    def test_malformed_entry_raises(self, tmp_path):
        """A malformed line fails loudly rather than silently losing a meeting."""
        path = tmp_path / "bad.yaml"
        path.write_text("meetings:\n  - meeting_id: x\n    this line has no colon\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_hand_counts(str(path))

    def test_parses_values_and_comments(self, tmp_path):
        """Inline comments are stripped and values typed."""
        path = tmp_path / "ok.yaml"
        path.write_text(
            "# header\nmeetings:\n  - meeting_id: 2011-06-08:work_study  # a comment\n"
            "    page_count: 5\n    motions_total: null\n"
            "  - meeting_id: 2025-02-26:regular\n    motions_total: 13\n",
            encoding="utf-8",
        )
        got = load_hand_counts(str(path))
        assert len(got) == 2
        assert got[0]["meeting_id"] == "2011-06-08:work_study"
        assert got[0]["page_count"] == 5
        assert got[0]["motions_total"] is None
        assert got[1]["motions_total"] == 13


# Faithful excerpt of agenda item 69513d40-ae2b-48a2-98e3-a9b72a6cab20,
# "Director District No. 4 Finalist - Discussion, Roll Call Vote, and
# Appointment", from the 2025-02-11 special meeting. One motion, followed by
# nomination roll-call rounds that are NOT votes on that motion.
NOMINATION_ROLL_CALL_ITEM = """Agenda Item Details
Motion & Voting
View All Motions
The board completed their interviews earlier than anticipated prior to the
executive session scheduled at 9:25 p.m.
A motion was made to approve the aforementioned schedule.
Motion by Tim Clark, second by Donald Cook.
Final Resolution: Motion Carries
Yea: Tim Clark, Meghin Margel, Donald Cook, Andy Song
The process to select the new Director District 4 board position from the
interview finalists took place via nominations and roll call voting as follows:
Nominees: Teresa Gregory, Thomas Foege, David Stanford
ROLL CALL VOTING ROUND 1
Nominated Finalist #1 Teresa Gregory
Yea: Song
Nay: Clark, Cook, Margel
Nomination Fails
Nominated Finalist #2 Thomas Foege
Yea: Clark
Nay: Song, Cook, Margel
Nomination Fails
ROLL CALL VOTING ROUND 2
Nominated Finalist #1 Teresa Gregory
Yea: Song, Clark
Nay: Cook, Margel
Nomination Fails
"""


class TestNominationRollCallDoesNotBleed:
    """A nomination roll-call sequence is not a vote on the preceding motion.

    Regression test for the 2025-02-11 special meeting, which recorded 8 votes
    from a 4-member board. The motion has one ``Final Resolution:`` anchor and
    nothing after it to stop at, so the tail scan ran to end-of-document and
    swept in every nomination round.
    """

    def test_only_the_canonical_roll_is_attributed_to_the_motion(self):
        """The motion gets its own 4 votes, not the nomination rounds."""
        motions = parse_agenda_item(NOMINATION_ROLL_CALL_ITEM)
        assert len(motions) == 1
        motion = motions[0]
        assert [v.director_raw for v in motion.votes] == [
            "Tim Clark",
            "Meghin Margel",
            "Donald Cook",
            "Andy Song",
        ]

    def test_tally_cannot_exceed_a_four_member_board(self):
        """The defect's signature: 5 yes + 3 no = 8 votes from 4 directors."""
        motion = parse_agenda_item(NOMINATION_ROLL_CALL_ITEM)[0]
        cast = (motion.tally_yes or 0) + (motion.tally_no or 0) + (motion.tally_abstain or 0)
        assert cast == 4, f"expected 4 votes, got {cast} (roll bleed)"
        assert motion.tally_yes == 4
        assert motion.tally_no == 0

    def test_surname_only_nomination_names_are_not_recorded(self):
        """'Song' from a nomination round must not join 'Andy Song'."""
        motion = parse_agenda_item(NOMINATION_ROLL_CALL_ITEM)[0]
        names = [v.director_raw for v in motion.votes]
        assert "Song" not in names
        assert "Clark" not in names
        assert "Margel" not in names
        assert len(names) == len(set(names))

    def test_a_later_heading_bounds_the_block(self):
        """A following Motion & Voting / Recommended Action ends the block."""
        text = NOMINATION_ROLL_CALL_ITEM.replace("The process to select", "Recommended Action\nThe process to select")
        motion = parse_agenda_item(text)[0]
        assert (motion.tally_yes or 0) + (motion.tally_no or 0) == 4

    def test_multi_motion_item_is_unaffected(self):
        """The ordinary two-motion shape still parses exactly as before."""
        motions = parse_agenda_item(AGENDA_ITEM)
        assert len(motions) == 2
        assert motions[0].tally_yes == 5
        assert motions[0].tally_no == 0
        assert motions[1].tally_yes == 1
        assert motions[1].tally_no == 3
        assert motions[1].tally_abstain == 1


class TestAgendaMotionQuoteReachesDisposition:
    """The locator quote must contain the disposition it claims."""

    def test_quote_reaches_final_resolution(self):
        """A short motion's quote spans opening to resolution."""
        motion = parse_agenda_item(AGENDA_ITEM)[0]
        assert "Final Resolution: Motion Carries" in motion.quote

    def test_long_motion_quote_still_reaches_disposition(self):
        """A consent motion longer than the old 400-char cap still reaches it."""
        filler = "and the item " * 200
        text = AGENDA_ITEM.replace(
            "Resolution No. 1697 - District Budget Adoption.",
            "Resolution No. 1697 " + filler + " District Budget Adoption.",
        )
        motion = parse_agenda_item(text)[0]
        assert len(motion.quote) > 400
        assert "Final Resolution: Motion Carries" in motion.quote

    def test_cap_clears_the_longest_span_in_the_corpus(self):
        """The cap is set above the longest measured motion-to-resolution span."""
        # Imported here, not at module scope: the roll-bleed regression tests
        # above must fail on the OLD parser with an assertion, and a top-level
        # import of a constant the old parser lacks would mask that with an
        # ImportError instead.
        from vote_parser import MOTION_QUOTE_MAX_LEN

        assert MOTION_QUOTE_MAX_LEN >= 3518
