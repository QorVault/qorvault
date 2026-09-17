-- Fact layer for Kent School District voucher and warrant registers.
--
-- Design notes the column list alone will not tell you:
--
--  * A row's locator is (file_path, file_sha256, page, char_offset, quote).
--    locator_document_id is NULLABLE and is never guessed. 44 voucher PDFs
--    exist on disk with no documents row at all -- including every file in
--    both 2026 sets that carry the hard fixtures -- so a NOT NULL foreign
--    key to documents would have excluded exactly the evidence the board
--    most needs. The digest is the durable half: a path changes when a
--    directory is renamed (which has already happened once in this corpus),
--    the bytes do not. relink.py fills document ids in later, keyed on the
--    digest, and is idempotent.
--
--  * source distinguishes a scraped file from one the operator staged. A
--    voucher packet is published on meeting night and the scrape that would
--    collect it runs later, so staged_pdf is a normal state, not an error
--    state, and it is not a lower grade of evidence.
--
--  * reconciled is NULLABLE, and the three states are distinct:
--      true   lines sum to a stated total, to the cent
--      false  lines sum to something else -- delta and reason_code say what
--      NULL   there is nothing to reconcile against
--    303 detail listings from 2017-2022 print no TOTAL line at all. Marking
--    those false would assert the district's arithmetic is wrong when what
--    is actually true is that the document states no arithmetic.
--
--  * fund has nine values, not five. Transportation Vehicle, Custodial and
--    Permanent funds pay real district money -- a five-value constraint
--    would reject 40 real sets and report a school bus as never bought.
--
--  * ACH is a payment METHOD, not a fund. The 2026-03-25 ACH listing total
--    equals the sum of the accounts-payable direct-deposit lines across
--    General, Capital, ASB and Custodial in the signed register, exactly.
--    It is modelled as a fund only because the district publishes it as its
--    own listing with its own TOTAL.
--
--  * voucher_reconciliation is a stored record, not a log line. The signed
--    register is independent evidence produced from a different query, and
--    where it disagrees with the listing that disagreement is a civic fact.
--    On 2026-03-25 two ASB warrants totalling $581.50 appear on the board's
--    signed register and not in the listing the public is given.
--
--  * check_amount repeats across every invoice line of a multi-invoice
--    check. sum(invoice_amount) is the figure the printed TOTAL equals;
--    sum(deduped check_amount) is a second control total that legitimately
--    differs on P-card and credit rows and is stored, not asserted.

CREATE SCHEMA IF NOT EXISTS facts;

-- ------------------------------------------------------------ voucher_set --
CREATE TABLE IF NOT EXISTS facts.voucher_set (
    set_id                   text PRIMARY KEY,
    tenant_id                varchar(64) NOT NULL DEFAULT 'kent_sd',
    meeting_date             date NOT NULL,
    fund                     text NOT NULL
        CHECK (fund IN ('GF', 'ACH', 'Capital', 'ASB', 'Trust',
                        'Transportation', 'Custodial', 'Permanent')),

    -- Provenance of the listing itself.
    source_document_id       uuid REFERENCES documents(id),
    agenda_item_document_id  uuid REFERENCES documents(id),
    doc_class                text NOT NULL
        CHECK (doc_class IN ('detail_listing', 'warrant_recap',
                             'warrant_register', 'voucher_by_vendor')),
    format_era               text CHECK (format_era IN ('A', 'B', 'C', 'D')),

    -- Washington State Archives disposition authority for school district
    -- accounts-payable records. Stored per row so an export can cite it.
    dan                      text NOT NULL DEFAULT 'GS2011-184',

    -- Periods as printed in the listing header.
    period_start             date,
    period_end               date,
    pcard_period_start       date,
    pcard_period_end         date,

    -- Three control totals, all stored.
    stated_total             numeric(14,2),
    parsed_total             numeric(14,2),
    sum_check_dedup          numeric(14,2),
    line_count               integer NOT NULL DEFAULT 0,
    check_count              integer NOT NULL DEFAULT 0,
    hash_total               bigint,

    reconciled               boolean,
    delta                    numeric(14,2),
    -- TOTAL_INCONSISTENT_AT_SOURCE is deliberately not OUT_OF_BALANCE.
    -- OUT_OF_BALANCE says the parse and the document disagree and the parse
    -- is the thing to go and check. This code says the document disagrees
    -- with itself -- its printed TOTAL cannot be produced by the rows
    -- printed beneath it under any reading of them. Filing the two under
    -- one code would send an auditor to re-read a parse that is correct,
    -- and would hide a class of finding the board is entitled to see.
    reason_code              text
        CHECK (reason_code IN ('OUT_OF_BALANCE', 'MULTIPLE_TOTALS',
                               'TOTAL_NOT_FOUND', 'NO_TEXT_LAYER',
                               'REGEX_MISS', 'DUPLICATE_SET',
                               'COLUMN_AMBIGUOUS',
                               'TOTAL_INCONSISTENT_AT_SOURCE', 'OTHER')),
    notes                    text,

    source                   text NOT NULL
        CHECK (source IN ('corpus_pdf', 'staged_pdf')),

    -- Locator of the TOTAL line this set reconciles against.
    locator_document_id      uuid REFERENCES documents(id),
    locator_file_path        text NOT NULL,
    locator_file_sha256      text NOT NULL,
    locator_page             integer,
    locator_char_offset      integer,
    locator_quote            text,

    created_at               timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_date, fund, locator_file_sha256)
);

-- ----------------------------------------------------------- voucher_line --
CREATE TABLE IF NOT EXISTS facts.voucher_line (
    line_id              bigserial PRIMARY KEY,
    set_id               text NOT NULL REFERENCES facts.voucher_set(set_id)
                              ON DELETE CASCADE,
    line_seq             integer NOT NULL,

    vendor_raw           text NOT NULL,
    vendor_norm          text NOT NULL,
    check_date           date,
    check_number         text,
    check_amount         numeric(14,2),
    invoice_amount       numeric(14,2),
    description          text,

    -- P-card pseudo-checks begin 926; payroll warrants are 530xxx. Both are
    -- flags on a real row, not a different kind of row.
    is_pcard             boolean NOT NULL DEFAULT false,
    is_payroll_warrant   boolean NOT NULL DEFAULT false,
    -- "Surname, Given" shape. Drives the export's personal-name rule; the
    -- row itself stays in the table.
    is_person_shaped     boolean NOT NULL DEFAULT false,
    is_credit            boolean NOT NULL DEFAULT false,

    -- A row the column grid could not read is kept, not dropped. The table
    -- must be able to say "this row is on the page and these are the
    -- columns that would not parse", because a row that is printed and
    -- absent here is a silent loss of public money. Exactly one code:
    -- every way a row can fail column assignment is the same kind of fact,
    -- and reason_detail carries which column broke.
    reason_code          text CHECK (reason_code IN ('COLUMN_AMBIGUOUS')),
    reason_detail        text,
    -- What the retired regex row parser made of the same line. Geometry is
    -- authoritative; this is a cross-check, never an input.
    regex_verdict        text CHECK (regex_verdict IN ('agree', 'disagree', 'regex_miss')),

    source               text NOT NULL
        CHECK (source IN ('corpus_pdf', 'staged_pdf')),

    locator_document_id  uuid REFERENCES documents(id),
    locator_file_path    text NOT NULL,
    locator_file_sha256  text NOT NULL,
    locator_page         integer,
    locator_char_offset  integer NOT NULL,
    locator_quote        text NOT NULL,

    UNIQUE (set_id, line_seq)
);

-- ---------------------------------------------------------------- vendor --
-- vendor_norm is deterministic: trim, collapse whitespace, strip trailing
-- punctuation, casefold. Exact and case-insensitive only. Corporate
-- suffixes are NOT stripped and abbreviations are NOT expanded -- every
-- such rule is a guess, and a guess that merges two vendors cannot be
-- undone once the rows are written.
CREATE TABLE IF NOT EXISTS facts.vendor (
    vendor_norm                  text PRIMARY KEY,
    display_name                 text NOT NULL,
    aliases                      text[] NOT NULL DEFAULT '{}',

    -- The only column an LLM may ever write, and only with tag_source='llm'.
    -- Nothing downstream of an amount, date, vendor identity, check number
    -- or total depends on it.
    category                     text,
    tag_source                   text CHECK (tag_source IN ('manual', 'llm')),

    is_person_shaped             boolean NOT NULL DEFAULT false,
    first_seen                   date,
    last_seen                    date,
    line_count                   integer NOT NULL DEFAULT 0,
    total_invoice                numeric(16,2),

    observed_locator_document_id uuid REFERENCES documents(id),
    observed_locator_file_path   text,
    observed_locator_file_sha256 text,
    observed_locator_page        integer,
    observed_locator_char_offset integer,
    observed_locator_quote       text
);

-- ------------------------------------------------ voucher_reconciliation --
-- One row per (set, basis). The basis says WHAT was compared, because the
-- register and the listing are different scopes and a single "does it
-- match" boolean would be meaningless without it.
CREATE TABLE IF NOT EXISTS facts.voucher_reconciliation (
    reconciliation_id        bigserial PRIMARY KEY,
    set_id                   text NOT NULL REFERENCES facts.voucher_set(set_id)
                                  ON DELETE CASCADE,

    basis                    text NOT NULL
        CHECK (basis IN ('ap_direct_deposit',      -- ACH listing vs register DD lines
                         'warrants_plus_pcard',    -- fund listing vs register AP + P-card
                         'recap_fund_total')),     -- listing vs Warrant Recap fund total

    register_document_id     uuid REFERENCES documents(id),
    register_file_path       text,
    register_file_sha256     text,
    register_total           numeric(14,2),
    register_warrant_range   text,
    detail_total             numeric(14,2),
    match                    boolean,
    delta                    numeric(14,2),
    reason                   text
        CHECK (reason IN ('MATCH', 'REGISTER_MISMATCH', 'REGISTER_NOT_FOUND',
                          'REGISTER_NO_TEXT', 'SCOPE_DIFFERS')),
    notes                    text,

    locator_detail_page         integer,
    locator_detail_char_offset  integer,
    locator_detail_quote        text,
    locator_register_page       integer,
    locator_register_char_offset integer,
    locator_register_quote      text,

    UNIQUE (set_id, basis)
);

-- ------------------------------------------------------------- parse log --
-- A failure rate by year and by reason is a deliverable, so every artifact
-- attempted gets a row whether or not it produced a set.
CREATE TABLE IF NOT EXISTS facts.voucher_parse_log (
    parse_id             bigserial PRIMARY KEY,
    file_sha256          text NOT NULL,
    file_path            text NOT NULL,
    document_id          uuid REFERENCES documents(id),
    meeting_date         date,
    fund                 text,
    doc_class            text,
    format_era           text,
    status               text NOT NULL
        CHECK (status IN ('parsed', 'no_text_layer', 'unreadable',
                          'era_unmatched', 'regex_miss', 'total_not_found',
                          'out_of_balance', 'column_ambiguous', 'skipped')),
    pages                integer,
    lines_found          integer NOT NULL DEFAULT 0,
    checks_found         integer NOT NULL DEFAULT 0,
    note                 text,
    -- How the columns were read, and how the retired regex path scored
    -- against them. Kept per artifact so a disagreement rate by set is a
    -- query rather than a rerun.
    grid_schema          text,
    grid_method          text CHECK (grid_method IN ('runs', 'header_band')),
    unread_lines         integer NOT NULL DEFAULT 0,
    regex_agree          integer NOT NULL DEFAULT 0,
    regex_disagree       integer NOT NULL DEFAULT 0,
    regex_miss           integer NOT NULL DEFAULT 0,
    source               text NOT NULL
        CHECK (source IN ('corpus_pdf', 'staged_pdf')),
    parsed_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (file_sha256)
);

CREATE INDEX IF NOT EXISTS idx_vset_date       ON facts.voucher_set(meeting_date);
CREATE INDEX IF NOT EXISTS idx_vset_fund       ON facts.voucher_set(fund);
CREATE INDEX IF NOT EXISTS idx_vset_recon      ON facts.voucher_set(reconciled);
CREATE INDEX IF NOT EXISTS idx_vset_sha        ON facts.voucher_set(locator_file_sha256);
CREATE INDEX IF NOT EXISTS idx_vline_set       ON facts.voucher_line(set_id);
CREATE INDEX IF NOT EXISTS idx_vline_vendor    ON facts.voucher_line(vendor_norm);
CREATE INDEX IF NOT EXISTS idx_vline_check     ON facts.voucher_line(check_number);
CREATE INDEX IF NOT EXISTS idx_vline_sha       ON facts.voucher_line(locator_file_sha256);
-- No trigram index on description. pg_trgm is available but not installed,
-- and CREATE EXTENSION is a database-wide change outside schema facts --
-- out of bounds for this build without the operator asking for it. A
-- sequential regex scan over this many lines is milliseconds; if
-- description_search ever becomes slow, installing pg_trgm is the fix and
-- it is a deliberate decision, not a side effect of running schema.sql.
CREATE INDEX IF NOT EXISTS idx_vrecon_set      ON facts.voucher_reconciliation(set_id);
CREATE INDEX IF NOT EXISTS idx_vplog_date      ON facts.voucher_parse_log(meeting_date);


-- ---------------------------------------------------------- R1 migration --
-- Applied to a database built before column assignment moved to geometry.
-- CREATE TABLE IF NOT EXISTS above does nothing to an existing table, so
-- the same changes are repeated here as idempotent ALTERs. Safe to run
-- against a fresh database too: every statement is a no-op there.

ALTER TABLE facts.voucher_line
    ADD COLUMN IF NOT EXISTS reason_code   text,
    ADD COLUMN IF NOT EXISTS reason_detail text,
    ADD COLUMN IF NOT EXISTS regex_verdict text;

-- ---------------------------------------------------- close-out migration --
-- Parentheses around an amount are accounting notation for a negative
-- number. Recording WHICH rows are written that way, rather than only the
-- resulting sign, is what lets the operator ask a later question -- "show me
-- every row whose sign came from a bracket" -- without re-reading 459 PDFs.
-- The sign itself lives in check_amount / invoice_amount as it always has.
ALTER TABLE facts.voucher_line
    ADD COLUMN IF NOT EXISTS amount_paren boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN facts.voucher_line.amount_paren IS
    'The amount column was printed in parentheses, i.e. the sign is from accounting notation.';

CREATE INDEX IF NOT EXISTS idx_vline_paren ON facts.voucher_line(set_id)
    WHERE amount_paren;

ALTER TABLE facts.voucher_parse_log
    ADD COLUMN IF NOT EXISTS grid_schema    text,
    ADD COLUMN IF NOT EXISTS grid_method    text,
    ADD COLUMN IF NOT EXISTS unread_lines   integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS regex_agree    integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS regex_disagree integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS regex_miss     integer NOT NULL DEFAULT 0;

DO $$
BEGIN
    ALTER TABLE facts.voucher_line  DROP CONSTRAINT IF EXISTS voucher_line_reason_code_check;
    ALTER TABLE facts.voucher_line  DROP CONSTRAINT IF EXISTS voucher_line_regex_verdict_check;
    ALTER TABLE facts.voucher_set   DROP CONSTRAINT IF EXISTS voucher_set_reason_code_check;
    ALTER TABLE facts.voucher_parse_log DROP CONSTRAINT IF EXISTS voucher_parse_log_status_check;
    ALTER TABLE facts.voucher_parse_log DROP CONSTRAINT IF EXISTS voucher_parse_log_grid_method_check;

    ALTER TABLE facts.voucher_line ADD CONSTRAINT voucher_line_reason_code_check
        CHECK (reason_code IN ('COLUMN_AMBIGUOUS'));
    ALTER TABLE facts.voucher_line ADD CONSTRAINT voucher_line_regex_verdict_check
        CHECK (regex_verdict IN ('agree', 'disagree', 'regex_miss'));
    ALTER TABLE facts.voucher_set ADD CONSTRAINT voucher_set_reason_code_check
        CHECK (reason_code IN ('OUT_OF_BALANCE', 'MULTIPLE_TOTALS', 'TOTAL_NOT_FOUND',
                               'NO_TEXT_LAYER', 'REGEX_MISS', 'DUPLICATE_SET',
                               'COLUMN_AMBIGUOUS', 'TOTAL_INCONSISTENT_AT_SOURCE',
                               'OTHER'));
    ALTER TABLE facts.voucher_parse_log ADD CONSTRAINT voucher_parse_log_status_check
        CHECK (status IN ('parsed', 'no_text_layer', 'unreadable', 'era_unmatched',
                          'regex_miss', 'total_not_found', 'out_of_balance',
                          'column_ambiguous', 'total_inconsistent_at_source',
                          'skipped'));
    ALTER TABLE facts.voucher_parse_log ADD CONSTRAINT voucher_parse_log_grid_method_check
        CHECK (grid_method IN ('runs', 'header_band'));
END $$;

CREATE INDEX IF NOT EXISTS idx_vline_reason ON facts.voucher_line(reason_code)
    WHERE reason_code IS NOT NULL;
