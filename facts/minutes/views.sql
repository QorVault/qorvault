-- Query surface over the minutes fact tables.
--
-- Every view exposes the locator columns so any figure can be traced back to a
-- source document, page and verbatim quote.

-- Motions in date order, with their meeting context and vote coverage.
CREATE OR REPLACE VIEW facts.motions_by_date AS
SELECT m.meeting_date,
       m.meeting_type,
       mo.motion_id,
       mo.motion_seq,
       mo.motion_number_raw,
       mo.motion_text,
       mo.disposition,
       mo.vote_format,
       mo.is_consent_agenda,
       mo.mover_raw,
       mo.second_raw,
       mo.tally_yes,
       mo.tally_no,
       mo.tally_abstain,
       mo.source,
       count(v.vote_id) AS named_votes,
       mo.locator_document_id,
       mo.locator_page,
       mo.locator_char_offset,
       mo.locator_quote
FROM facts.motion mo
JOIN facts.meeting m USING (meeting_id)
LEFT JOIN facts.vote v USING (motion_id)
GROUP BY m.meeting_date, m.meeting_type, mo.motion_id, mo.motion_seq,
         mo.motion_number_raw, mo.motion_text, mo.disposition, mo.vote_format,
         mo.is_consent_agenda, mo.mover_raw, mo.second_raw, mo.tally_yes,
         mo.tally_no, mo.tally_abstain, mo.source, mo.locator_document_id,
         mo.locator_page, mo.locator_char_offset, mo.locator_quote;

-- One row per director per vote. director_norm is NULL corpus-wide because no
-- roster exists, so callers should group on director_raw.
CREATE OR REPLACE VIEW facts.votes_by_director AS
SELECT coalesce(v.director_norm, v.director_raw) AS director,
       v.director_raw,
       v.director_norm,
       m.meeting_date,
       m.meeting_type,
       mo.motion_id,
       mo.motion_text,
       mo.disposition,
       v.vote,
       v.source,
       v.locator_document_id,
       v.locator_page,
       v.locator_char_offset,
       v.locator_quote
FROM facts.vote v
JOIN facts.motion mo USING (motion_id)
JOIN facts.meeting m ON m.meeting_id = mo.meeting_id;

-- Executive sessions per year, counted two ways because the corpus records
-- them two ways:
--   scheduled_meetings -- executive sessions convened as their own meeting,
--                         from the BoardDocs meeting census
--   announcements      -- executive sessions announced inside another
--                         meeting's minutes (one row per announcement, so an
--                         extension counts separately)
-- The two are disjoint and both are reported; "total" is their sum.
CREATE OR REPLACE VIEW facts.exec_sessions_by_year AS
WITH scheduled AS (
    SELECT extract(year FROM meeting_date)::int AS yr, count(*) AS n
    FROM facts.meeting
    WHERE meeting_type = 'exec_session'
    GROUP BY 1
),
announced AS (
    SELECT extract(year FROM m.meeting_date)::int AS yr, count(*) AS n
    FROM facts.executive_session e
    JOIN facts.meeting m USING (meeting_id)
    WHERE m.meeting_type <> 'exec_session'
    GROUP BY 1
)
SELECT coalesce(s.yr, a.yr) AS year,
       coalesce(s.n, 0) AS scheduled_meetings,
       coalesce(a.n, 0) AS announcements,
       coalesce(s.n, 0) + coalesce(a.n, 0) AS total
FROM scheduled s
FULL OUTER JOIN announced a ON s.yr = a.yr
ORDER BY 1;

-- Meetings the district held for which no minutes document exists in the
-- corpus. Answerable only because facts.meeting is seeded from the meeting
-- census rather than from the minutes themselves.
CREATE OR REPLACE VIEW facts.meetings_missing_minutes AS
SELECT meeting_id,
       meeting_date,
       meeting_type,
       census_slug,
       source
FROM facts.meeting
WHERE minutes_document_id IS NULL
ORDER BY meeting_date, meeting_type;

-- Motions where more directors voted than the minutes record as present.
--
-- This is a data-quality surface, not an error list. The check that used to
-- assert "this never happens" now asserts something narrower and true: it
-- never happens OUTSIDE a maintained known-set. Each known meeting carries the
-- reason it differs, so a reader sees the cause rather than a bare count.
--
-- cause vocabulary:
--   board_transition  -- the minutes record one board's roll and the votes
--                        were cast by another (a seating meeting). Both
--                        records are accurate; they describe different boards.
--   attendance_short  -- the attendance list in the minutes omits a director
--                        who is recorded voting later in the same meeting.
--   presiding_only    -- the minutes record only the presiding officer, not
--                        the full roll.
--   status_excluded   -- a director marked absent/excused in the roll is
--                        nonetheless recorded voting, so the "present" count
--                        legitimately falls below the number of voters.
--   NULL              -- a case not in the known-set. A NULL cause is a new
--                        discrepancy and fails the hard fixture.
--
-- Every cause above describes the DISTRICT'S record, not our parsing of it. A
-- sixth entry, 2025-02-11, was briefly carried as `parser_roll_bleed` while a
-- nomination roll-call sequence was being attributed to the preceding motion
-- (8 votes from a 4-member board). That was our bug, not theirs; it was fixed
-- in vote_parser._canonical_roll_block and the meeting left this view.
CREATE OR REPLACE VIEW facts.attendance_vote_discrepancies AS
WITH present AS (
    SELECT meeting_id, count(*) AS n
    FROM facts.attendance
    WHERE status IN ('present', 'present_virtual', 'arrived_late', 'left_early')
    GROUP BY meeting_id
),
known (meeting_id, cause) AS (
    VALUES ('2022-06-29:special', 'presiding_only'),
           ('2022-10-05:special', 'attendance_short'),
           ('2023-11-08:regular', 'status_excluded'),
           ('2023-12-13:regular', 'board_transition'),
           ('2024-07-10:special', 'status_excluded')
)
SELECT m.meeting_date,
       m.meeting_type,
       mo.meeting_id,
       mo.motion_id,
       mo.source,
       p.n AS present_recorded,
       coalesce(mo.tally_yes, 0) + coalesce(mo.tally_no, 0)
         + coalesce(mo.tally_abstain, 0) AS cast_votes,
       k.cause,
       mo.locator_document_id,
       mo.locator_page,
       mo.locator_char_offset,
       mo.locator_quote
FROM facts.motion mo
JOIN present p ON p.meeting_id = mo.meeting_id
JOIN facts.meeting m ON m.meeting_id = mo.meeting_id
LEFT JOIN known k ON k.meeting_id = mo.meeting_id
WHERE (mo.tally_yes IS NOT NULL
       OR mo.tally_no IS NOT NULL
       OR mo.tally_abstain IS NOT NULL)
  AND coalesce(mo.tally_yes, 0) + coalesce(mo.tally_no, 0)
      + coalesce(mo.tally_abstain, 0) > p.n
ORDER BY m.meeting_date, mo.motion_id;

-- How much of the motion record carries no director names at all. This is the
-- headline transparency measure: for most of the corpus the minutes record
-- only "Motion carried."
CREATE OR REPLACE VIEW facts.votes_unnamed_by_year AS
SELECT extract(year FROM m.meeting_date)::int AS year,
       count(*) AS motions,
       count(*) FILTER (WHERE mo.vote_format = 'named') AS motions_named,
       count(*) FILTER (WHERE mo.vote_format <> 'named') AS motions_unnamed,
       round(100.0 * count(*) FILTER (WHERE mo.vote_format <> 'named')
             / nullif(count(*), 0), 1) AS pct_unnamed,
       count(*) FILTER (WHERE mo.source = 'minutes') AS from_minutes,
       count(*) FILTER (WHERE mo.source = 'agenda_item') AS from_agenda_items
FROM facts.motion mo
JOIN facts.meeting m USING (meeting_id)
GROUP BY 1
ORDER BY 1;
