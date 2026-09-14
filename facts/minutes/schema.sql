-- Fact layer for Kent School District board meeting minutes.
--
-- Design notes that the column list alone will not tell you:
--
--  * meeting.meeting_date is the date stated IN THE MINUTES BODY, never
--    documents.meeting_date. Minutes are attached to the agenda of the later
--    meeting that approves them (Phase 0: 796/796 offsets positive, 88.6%
--    exact multiples of 7 days). documents.meeting_date therefore feeds
--    approved_at_meeting_id, not meeting_date. Reversing these two misdates
--    the entire fact layer by one meeting.
--
--  * Every fact row carries a locator: source document_id, page number, and a
--    verbatim quote plus character offset into the extracted text. Chunk ids
--    are never used as a locator.
--
--  * locator_page is NULLABLE. Phase 0 established that Postgres holds no
--    page-level text at all (document_pages empty, chunks.source_page 100%
--    NULL), so page numbers come from re-reading the source PDFs. Where that
--    is unavailable the character offset still resolves the quote.
--
--  * vote rows exist only where names actually appear. In the minutes they
--    almost never do (4 of 874 documents). The named votes come from BoardDocs
--    agenda_item documents, 2018-2026 only; vote.source is set accordingly.
--
--  * director_norm is NULL throughout: no director roster exists anywhere in
--    the repo or the database. Raw names only, no fuzzy matching.

CREATE SCHEMA IF NOT EXISTS facts;

-- ---------------------------------------------------------------- meeting --
CREATE TABLE IF NOT EXISTS facts.meeting (
    meeting_id              text PRIMARY KEY,
    tenant_id               varchar(64)  NOT NULL DEFAULT 'kent_sd',
    meeting_date            date         NOT NULL,
    meeting_type            text         NOT NULL
        CHECK (meeting_type IN ('regular', 'special', 'work_study',
                                'exec_session', 'other')),
    minutes_document_id     uuid         REFERENCES documents(id),
    approved_at_meeting_id  text,
    approved_as_corrected   boolean      NOT NULL DEFAULT false,
    correction_note         text,
    format_era              text         CHECK (format_era IN ('A', 'B')),
    locator_document_id     uuid         REFERENCES documents(id),
    locator_page            integer,
    locator_char_offset     integer,
    locator_quote           text,
    created_at              timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (meeting_date, meeting_type, minutes_document_id)
);

-- ------------------------------------------------------------- attendance --
CREATE TABLE IF NOT EXISTS facts.attendance (
    attendance_id        bigserial PRIMARY KEY,
    meeting_id           text NOT NULL REFERENCES facts.meeting(meeting_id)
                              ON DELETE CASCADE,
    director_raw         text NOT NULL,
    director_norm        text,
    role_raw             text,
    status               text NOT NULL
        CHECK (status IN ('present', 'absent', 'arrived_late', 'left_early',
                          'present_virtual', 'excused')),
    locator_document_id  uuid REFERENCES documents(id),
    locator_page         integer,
    locator_char_offset  integer,
    locator_quote        text,
    UNIQUE (meeting_id, director_raw)
);

-- ----------------------------------------------------------------- motion --
CREATE TABLE IF NOT EXISTS facts.motion (
    motion_id            text PRIMARY KEY,
    meeting_id           text NOT NULL REFERENCES facts.meeting(meeting_id)
                              ON DELETE CASCADE,
    motion_seq           integer NOT NULL,
    motion_number_raw    text,              -- e.g. "Motion No. 51-11" (era A)
    agenda_item_ref      text,              -- BoardDocs id when resolvable
    agenda_item_order    integer,           -- else position within the meeting
    motion_text          text NOT NULL,     -- verbatim, final wording
    pre_amendment_text   text,              -- original wording if amended
    mover_raw            text,
    second_raw           text,
    disposition          text NOT NULL
        CHECK (disposition IN ('adopted', 'lost', 'tabled', 'withdrawn')),
    tally_yes            integer,
    tally_no             integer,
    tally_abstain        integer,
    vote_format          text NOT NULL
        CHECK (vote_format IN ('named', 'roll_call', 'carried_no_names',
                               'tally_only')),
    is_consent_agenda    boolean NOT NULL DEFAULT false,
    source               text NOT NULL
        CHECK (source IN ('minutes', 'agenda_item')),
    locator_document_id  uuid REFERENCES documents(id),
    locator_page         integer,
    locator_char_offset  integer,
    locator_quote        text NOT NULL,
    UNIQUE (meeting_id, motion_seq, source)
);

-- ------------------------------------------------------------------- vote --
CREATE TABLE IF NOT EXISTS facts.vote (
    vote_id              bigserial PRIMARY KEY,
    motion_id            text NOT NULL REFERENCES facts.motion(motion_id)
                              ON DELETE CASCADE,
    director_raw         text NOT NULL,
    director_norm        text,
    vote                 text NOT NULL
        CHECK (vote IN ('yes', 'no', 'abstain', 'absent')),
    source               text NOT NULL
        CHECK (source IN ('minutes', 'agenda_item')),
    locator_document_id  uuid REFERENCES documents(id),
    locator_page         integer,
    locator_char_offset  integer,
    locator_quote        text,
    UNIQUE (motion_id, director_raw)
);

-- ------------------------------------------------------ executive_session --
-- One row per announcement. An extension or reconvening of a session is a
-- separate announcement and therefore a separate row.
CREATE TABLE IF NOT EXISTS facts.executive_session (
    exec_session_id      text PRIMARY KEY,
    meeting_id           text NOT NULL REFERENCES facts.meeting(meeting_id)
                              ON DELETE CASCADE,
    announcement_seq     integer NOT NULL,
    announced_purpose    text,              -- verbatim
    purpose_category     text,              -- RCW 42.30.110(1)(x) if named
    announced_at         text,              -- clock time as printed
    stated_end_time      text,
    actual_end_time      text,
    is_extension         boolean NOT NULL DEFAULT false,
    announcement_kind    text NOT NULL
        CHECK (announcement_kind IN ('scheduled_meeting', 'in_meeting',
                                     'extension', 'reconvene', 'adjourn')),
    locator_document_id  uuid REFERENCES documents(id),
    locator_page         integer,
    locator_char_offset  integer,
    locator_quote        text NOT NULL,
    UNIQUE (meeting_id, announcement_seq)
);

-- ------------------------------------------------------ parse diagnostics --
-- A failure rate by year is a deliverable, so failures are recorded as data
-- rather than only printed.
CREATE TABLE IF NOT EXISTS facts.minutes_parse_log (
    document_id          uuid PRIMARY KEY REFERENCES documents(id),
    minutes_date         date,
    format_era           text,
    status               text NOT NULL
        CHECK (status IN ('parsed', 'no_text_layer', 'era_unmatched',
                          'motion_unparsed', 'locator_unresolved',
                          'date_unresolved')),
    motions_found        integer NOT NULL DEFAULT 0,
    exec_sessions_found  integer NOT NULL DEFAULT 0,
    attendance_found     integer NOT NULL DEFAULT 0,
    note                 text,
    parsed_at            timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_meeting_date  ON facts.meeting(meeting_date);
CREATE INDEX IF NOT EXISTS idx_motion_meeting ON facts.motion(meeting_id);
CREATE INDEX IF NOT EXISTS idx_vote_motion    ON facts.vote(motion_id);
CREATE INDEX IF NOT EXISTS idx_vote_director  ON facts.vote(director_raw);
CREATE INDEX IF NOT EXISTS idx_exec_meeting   ON facts.executive_session(meeting_id);
CREATE INDEX IF NOT EXISTS idx_attend_meeting ON facts.attendance(meeting_id);
