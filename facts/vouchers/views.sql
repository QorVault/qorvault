-- Query surface for the voucher fact layer.
--
-- Every view carries the reconciliation status of the set a row came from.
-- That is deliberate: a figure from an unreconciled set is not the same
-- kind of fact as one from a set that ties to its printed total to the
-- cent, and a query surface that hides the difference invites someone to
-- say the wrong number out loud.
--
-- reconciled has three states everywhere:
--   true   lines sum to the printed total, to the cent
--   false  lines sum to something else (delta and reason_code say what)
--   NULL   the document prints no total to reconcile against

-- ------------------------------------------------------------ set_totals --
CREATE OR REPLACE VIEW facts.set_totals AS
SELECT s.set_id,
       s.meeting_date,
       s.fund,
       s.format_era,
       s.period_start,
       s.period_end,
       s.pcard_period_start,
       s.pcard_period_end,
       s.stated_total,
       s.parsed_total,
       s.sum_check_dedup,
       -- The second control total legitimately differs from the first on
       -- P-card and credit rows; the difference is reported, not asserted
       -- away.
       (s.parsed_total - s.sum_check_dedup)      AS invoice_minus_check,
       s.line_count,
       s.check_count,
       s.hash_total,
       s.reconciled,
       s.delta,
       s.reason_code,
       CASE
           WHEN s.reconciled IS TRUE  THEN 'reconciled'
           WHEN s.reconciled IS FALSE THEN 'FLAGGED: ' || COALESCE(s.reason_code, 'OTHER')
           ELSE 'no stated total in the document'
       END                                       AS status,
       s.notes,
       s.source,
       s.dan,
       s.locator_document_id,
       s.locator_file_path,
       s.locator_file_sha256,
       s.locator_page,
       s.locator_char_offset,
       s.locator_quote
FROM facts.voucher_set s;

COMMENT ON VIEW facts.set_totals IS
    'One row per voucher set with all three control totals and its reconciliation status.';

-- -------------------------------------------------------- vendor_by_cycle --
-- What a vendor was paid at each voucher night, per fund. This is the view
-- the watch list is built on.
CREATE OR REPLACE VIEW facts.vendor_by_cycle AS
SELECT s.meeting_date,
       s.fund,
       l.vendor_norm,
       v.display_name,
       v.is_person_shaped,
       count(*)                                  AS line_count,
       count(DISTINCT l.check_number)            AS check_count,
       sum(l.invoice_amount)                     AS invoice_total,
       min(l.check_date)                         AS first_check_date,
       max(l.check_date)                         AS last_check_date,
       bool_or(l.is_pcard)                       AS has_pcard,
       bool_or(l.is_credit)                      AS has_credit,
       s.reconciled,
       s.reason_code,
       -- One locator per (cycle, fund, vendor) so any figure here can be
       -- traced back to a page.
       (array_agg(l.locator_file_path ORDER BY l.line_seq))[1]   AS locator_file_path,
       (array_agg(l.locator_file_sha256 ORDER BY l.line_seq))[1] AS locator_file_sha256,
       (array_agg(l.locator_page ORDER BY l.line_seq))[1]        AS locator_page,
       (array_agg(l.locator_char_offset ORDER BY l.line_seq))[1] AS locator_char_offset,
       (array_agg(l.locator_quote ORDER BY l.line_seq))[1]       AS locator_quote
FROM facts.voucher_line l
JOIN facts.voucher_set  s ON s.set_id = l.set_id
LEFT JOIN facts.vendor  v ON v.vendor_norm = l.vendor_norm
GROUP BY s.meeting_date, s.fund, l.vendor_norm, v.display_name,
         v.is_person_shaped, s.reconciled, s.reason_code;

COMMENT ON VIEW facts.vendor_by_cycle IS
    'Vendor spend per voucher night per fund, with a locator and the set''s reconciliation status.';

-- ---------------------------------------------------- description_search --
-- Flat, searchable view of every line. Apply a regex to description in the
-- WHERE clause:
--
--   SELECT * FROM facts.description_search
--    WHERE description ~* 'legal|attorney' AND meeting_date >= '2025-09-01';
CREATE OR REPLACE VIEW facts.description_search AS
SELECT l.line_id,
       s.meeting_date,
       s.fund,
       l.vendor_raw,
       l.vendor_norm,
       l.check_date,
       l.check_number,
       l.check_amount,
       l.invoice_amount,
       l.description,
       l.is_pcard,
       l.is_payroll_warrant,
       l.is_credit,
       l.is_person_shaped,
       s.reconciled,
       s.reason_code,
       l.source,
       l.locator_document_id,
       l.locator_file_path,
       l.locator_file_sha256,
       l.locator_page,
       l.locator_char_offset,
       l.locator_quote
FROM facts.voucher_line l
JOIN facts.voucher_set  s ON s.set_id = l.set_id;

COMMENT ON VIEW facts.description_search IS
    'Every voucher line with its locator. Apply a regex to description in the WHERE clause.';

-- ---------------------------------------------------------- vendor_search --
CREATE OR REPLACE VIEW facts.vendor_search AS
SELECT v.vendor_norm,
       v.display_name,
       v.aliases,
       v.category,
       v.tag_source,
       v.is_person_shaped,
       v.first_seen,
       v.last_seen,
       v.line_count,
       v.total_invoice,
       (SELECT count(DISTINCT s2.meeting_date)
          FROM facts.voucher_line l2
          JOIN facts.voucher_set  s2 ON s2.set_id = l2.set_id
         WHERE l2.vendor_norm = v.vendor_norm)     AS cycles_seen,
       (SELECT count(DISTINCT s2.fund)
          FROM facts.voucher_line l2
          JOIN facts.voucher_set  s2 ON s2.set_id = l2.set_id
         WHERE l2.vendor_norm = v.vendor_norm)     AS funds_seen,
       -- Money from sets that reconcile is a different grade of evidence
       -- from money from sets that do not. Both are shown; neither is
       -- silently folded into the other.
       (SELECT COALESCE(sum(l3.invoice_amount), 0)
          FROM facts.voucher_line l3
          JOIN facts.voucher_set  s3 ON s3.set_id = l3.set_id
         WHERE l3.vendor_norm = v.vendor_norm
           AND s3.reconciled IS TRUE)              AS invoice_total_reconciled,
       (SELECT COALESCE(sum(l4.invoice_amount), 0)
          FROM facts.voucher_line l4
          JOIN facts.voucher_set  s4 ON s4.set_id = l4.set_id
         WHERE l4.vendor_norm = v.vendor_norm
           AND s4.reconciled IS DISTINCT FROM TRUE) AS invoice_total_unreconciled,
       v.observed_locator_file_path,
       v.observed_locator_file_sha256,
       v.observed_locator_page,
       v.observed_locator_char_offset,
       v.observed_locator_quote
FROM facts.vendor v;

COMMENT ON VIEW facts.vendor_search IS
    'One row per vendor, with spend split by whether its source set reconciles.';

-- ------------------------------------------------- reconciliation_by_year --
-- The failure rate by year and by reason is a deliverable, so it is a view
-- rather than a number in a report that goes stale.
CREATE OR REPLACE VIEW facts.reconciliation_by_year AS
SELECT extract(year FROM s.meeting_date)::int          AS year,
       count(*)                                        AS sets,
       count(*) FILTER (WHERE s.reconciled IS TRUE)    AS reconciled,
       count(*) FILTER (WHERE s.reconciled IS FALSE)   AS flagged,
       count(*) FILTER (WHERE s.reconciled IS NULL)    AS no_stated_total,
       round(100.0 * count(*) FILTER (WHERE s.reconciled IS TRUE)
             / NULLIF(count(*) FILTER (WHERE s.reconciled IS NOT NULL), 0), 1)
                                                       AS pct_reconciled_of_checkable,
       sum(s.parsed_total)                             AS parsed_total,
       sum(s.parsed_total) FILTER (WHERE s.reconciled IS TRUE)
                                                       AS parsed_total_reconciled
FROM facts.voucher_set s
GROUP BY 1
ORDER BY 1;

COMMENT ON VIEW facts.reconciliation_by_year IS
    'Reconciliation outcome by calendar year. pct is of sets that have a total to check.';

-- ----------------------------------------------- reconciliation_by_reason --
CREATE OR REPLACE VIEW facts.reconciliation_by_reason AS
SELECT COALESCE(s.reason_code, 'RECONCILED')           AS reason_code,
       count(*)                                        AS sets,
       min(s.meeting_date)                             AS first_seen,
       max(s.meeting_date)                             AS last_seen,
       sum(abs(COALESCE(s.delta, 0)))                  AS absolute_delta
FROM facts.voucher_set s
GROUP BY 1
ORDER BY 2 DESC;

COMMENT ON VIEW facts.reconciliation_by_reason IS
    'How many sets carry each reason code, and the total absolute delta behind it.';

-- ------------------------------------------------------- cumulative_sets --
-- Sets whose check numbers already appeared in an earlier set of the same
-- fund. Summing these with their predecessors double-counts real money.
CREATE OR REPLACE VIEW facts.cumulative_sets AS
SELECT s.set_id,
       s.meeting_date,
       s.fund,
       s.parsed_total,
       s.reconciled,
       s.notes
FROM facts.voucher_set s
WHERE s.notes LIKE 'cumulative:%' OR s.notes LIKE '%; cumulative:%'
ORDER BY s.fund, s.meeting_date;

COMMENT ON VIEW facts.cumulative_sets IS
    'Sets that restate an earlier cycle''s rows. Never sum these across meetings.';

-- --------------------------------------------------- register_crosschecks --
CREATE OR REPLACE VIEW facts.register_crosschecks AS
SELECT r.set_id,
       s.meeting_date,
       s.fund,
       r.basis,
       r.register_total,
       r.detail_total,
       r.delta,
       r.match,
       r.reason,
       r.register_warrant_range,
       r.notes,
       r.register_file_path,
       r.register_file_sha256,
       r.locator_register_page,
       r.locator_register_quote,
       s.locator_file_path                             AS detail_file_path,
       s.locator_page                                  AS detail_page,
       s.locator_quote                                 AS detail_quote
FROM facts.voucher_reconciliation r
JOIN facts.voucher_set s ON s.set_id = r.set_id
ORDER BY s.meeting_date, s.fund, r.basis;

COMMENT ON VIEW facts.register_crosschecks IS
    'Every listing checked against the board''s signed register, with both locators.';

-- ================================================== cross-cycle dedupe ==
-- Some listings restate earlier cycles. 145 sets across five funds carry
-- check numbers that already appeared in an earlier set of the same fund,
-- and summing those sets across meetings counts the same warrant twice.
-- facts.cumulative_sets says WHICH sets do it; these views say how much
-- money it is and give a total that is safe to add up across cycles.
--
-- The dedupe key is (fund, check_number). Check numbers are issued per
-- warrant series, so the same number in two different funds is two
-- different payments and must not be collapsed.
--
-- Rows that could not be read are excluded from the key: a row with a
-- reason code has no reliable check number to deduplicate on.
--
-- Sentinel numbers are excluded too, and for a sharper reason. A number
-- that is all one digit, or that carries a run of eight or more identical
-- digits, is not a warrant identifier: the accounting system emits it for
-- entries that have no warrant. Three such numbers are in this corpus --
-- 8888888888 and the pair 8888888898 / 8888888899. Keeping them in the key
-- would merge unrelated payments that happen to share a placeholder and
-- would delete real money from a cross-cycle total. Excluded from the key,
-- every sentinel row is its own payment and is always counted: the LEFT
-- JOIN below finds no first-cycle row for it, and is_first_cycle_for_check
-- is therefore true. This predicate is the SQL twin of is_sentinel_check()
-- in build.py; the two must say the same thing.

-- ------------------------------------------------------ check_first_cycle --
CREATE OR REPLACE VIEW facts.check_first_cycle AS
SELECT DISTINCT ON (s.fund, l.check_number)
       s.fund,
       l.check_number,
       s.meeting_date                          AS first_meeting_date,
       s.set_id                                AS first_set_id
FROM facts.voucher_line l
JOIN facts.voucher_set  s ON s.set_id = l.set_id
WHERE l.check_number IS NOT NULL
  AND l.reason_code IS NULL
  AND l.check_number !~ '^(.)\1*$'
  AND l.check_number !~ '(.)\1{7,}'
ORDER BY s.fund, l.check_number, s.meeting_date, s.set_id;

COMMENT ON VIEW facts.check_first_cycle IS
    'The earliest voucher night that printed each (fund, check number).';

-- ---------------------------------------------------- voucher_line_deduped --
CREATE OR REPLACE VIEW facts.voucher_line_deduped AS
SELECT l.line_id,
       l.set_id,
       s.meeting_date,
       s.fund,
       l.line_seq,
       l.vendor_raw,
       l.vendor_norm,
       l.check_date,
       l.check_number,
       l.check_amount,
       l.invoice_amount,
       l.description,
       l.is_pcard,
       l.is_payroll_warrant,
       l.is_person_shaped,
       l.is_credit,
       l.reason_code,
       l.reason_detail,
       -- True when this is the first voucher night to print this check, so
       -- the row may be added into a cross-cycle total. False means the
       -- money is real but was already counted at an earlier meeting.
       (f.first_set_id IS NULL OR f.first_set_id = l.set_id) AS is_first_cycle_for_check,
       f.first_meeting_date,
       s.reconciled,
       s.reason_code                            AS set_reason_code,
       l.source,
       l.locator_file_path,
       l.locator_file_sha256,
       l.locator_page,
       l.locator_char_offset,
       l.locator_quote
FROM facts.voucher_line l
JOIN facts.voucher_set  s ON s.set_id = l.set_id
LEFT JOIN facts.check_first_cycle f
       ON f.fund = s.fund AND f.check_number = l.check_number;

COMMENT ON VIEW facts.voucher_line_deduped IS
    'Every voucher line, flagged with whether its check first appears in this cycle.';

-- ------------------------------------------------------ set_totals_deduped --
CREATE OR REPLACE VIEW facts.set_totals_deduped AS
SELECT s.set_id,
       s.meeting_date,
       s.fund,
       s.stated_total,
       s.parsed_total,
       COALESCE(sum(l.invoice_amount) FILTER (
           WHERE l.reason_code IS NULL AND l.is_first_cycle_for_check), 0)  AS parsed_total_new_checks,
       COALESCE(sum(l.invoice_amount) FILTER (
           WHERE l.reason_code IS NULL AND NOT l.is_first_cycle_for_check), 0) AS restated_from_earlier,
       count(*) FILTER (WHERE NOT l.is_first_cycle_for_check)               AS restated_lines,
       count(DISTINCT l.check_number) FILTER (
           WHERE NOT l.is_first_cycle_for_check)                            AS restated_checks,
       count(*) FILTER (WHERE l.reason_code IS NOT NULL)                    AS unread_lines,
       s.line_count,
       s.check_count,
       s.reconciled,
       s.delta,
       s.reason_code
FROM facts.voucher_set s
LEFT JOIN facts.voucher_line_deduped l ON l.set_id = s.set_id
GROUP BY s.set_id, s.meeting_date, s.fund, s.stated_total, s.parsed_total,
         s.line_count, s.check_count, s.reconciled, s.delta, s.reason_code;

COMMENT ON VIEW facts.set_totals_deduped IS
    'Per set: the printed total, the parsed total, and how much of it was already counted at an earlier meeting.';

-- ------------------------------------------------ vendor_by_cycle_deduped --
CREATE OR REPLACE VIEW facts.vendor_by_cycle_deduped AS
SELECT l.meeting_date,
       l.fund,
       l.vendor_norm,
       v.display_name,
       v.is_person_shaped,
       count(*)                                                    AS line_count,
       count(DISTINCT l.check_number)                              AS check_count,
       sum(l.invoice_amount) FILTER (WHERE l.reason_code IS NULL)  AS invoice_total,
       sum(l.invoice_amount) FILTER (
           WHERE l.reason_code IS NULL AND l.is_first_cycle_for_check) AS invoice_total_new_checks,
       count(*) FILTER (WHERE l.reason_code IS NOT NULL)           AS unread_lines,
       l.reconciled,
       l.set_reason_code
FROM facts.voucher_line_deduped l
LEFT JOIN facts.vendor v ON v.vendor_norm = l.vendor_norm
GROUP BY l.meeting_date, l.fund, l.vendor_norm, v.display_name,
         v.is_person_shaped, l.reconciled, l.set_reason_code;

COMMENT ON VIEW facts.vendor_by_cycle_deduped IS
    'Vendor spend per cycle per fund, with the part already counted at an earlier meeting separated out.';

-- --------------------------------------------------- register_availability --
-- Whether a set has an independent register cross-check at all, as a fact
-- per set rather than as an absence a reader has to notice. The signed
-- registers for 2026-06-24 and 2026-07-22 are scans with no text layer, so
-- those two nights have no second opinion on this machine and an export
-- that did not say so would be overstating its evidence.
CREATE OR REPLACE VIEW facts.register_availability AS
SELECT s.set_id,
       s.meeting_date,
       s.fund,
       r.basis,
       r.reason,
       r.register_total,
       r.detail_total,
       r.delta,
       r.match,
       (r.reason IN ('MATCH', 'REGISTER_MISMATCH', 'SCOPE_DIFFERS')) AS has_crosscheck,
       CASE r.reason
           WHEN 'MATCH'              THEN 'checked against the signed register: ties'
           WHEN 'REGISTER_MISMATCH'  THEN 'checked against the signed register: differs'
           WHEN 'SCOPE_DIFFERS'      THEN 'checked against a Warrant Recap, which is a wider scope'
           WHEN 'REGISTER_NO_TEXT'   THEN 'no cross-check: the signed register is a scan with no text layer'
           WHEN 'REGISTER_NOT_FOUND' THEN 'no cross-check: no signed register or recap is on this machine'
           ELSE                           'no cross-check recorded'
       END                                                           AS crosscheck_note,
       r.register_file_path,
       r.locator_register_page,
       r.locator_register_quote
FROM facts.voucher_set s
LEFT JOIN facts.voucher_reconciliation r ON r.set_id = s.set_id;

COMMENT ON VIEW facts.register_availability IS
    'Per set: whether an independent register cross-check exists, and if not, why not.';

-- ----------------------------------------------------- unread_lines_by_set --
-- Rows that are printed on the page and could not be assigned to columns.
-- A deliverable in its own right: it is the list of everything this layer
-- knows it has not read.
CREATE OR REPLACE VIEW facts.unread_lines AS
SELECT s.meeting_date,
       s.fund,
       l.set_id,
       l.line_seq,
       l.reason_code,
       l.reason_detail,
       l.locator_file_path,
       l.locator_page,
       l.locator_char_offset,
       l.locator_quote
FROM facts.voucher_line l
JOIN facts.voucher_set  s ON s.set_id = l.set_id
WHERE l.reason_code IS NOT NULL
ORDER BY s.meeting_date, s.fund, l.line_seq;

COMMENT ON VIEW facts.unread_lines IS
    'Every printed row the column grid could not read, with the page it is on.';
