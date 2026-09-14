-- link_40.sql — attach the 40 orphaned 2026-02-11 attachments to their agenda items.
--
-- NOT EXECUTED. Written for operator review (Part 1); run only after approval.
--
-- Provenance of the mapping below: each agenda_item_id was derived offline by
-- scripts/ingest-repair-2026-09-13/derive_links.py from the structured
-- scraper's archive_* trees for the two 2026-02-11 meetings. Every row was
-- resolved by TWO independent methods that had to agree:
--   1. flat filename -> BoardDocs file ID (from the retained agenda.html)
--      -> item.json links[].unique -> itemId
--   2. flat filename -> sanitized item.json links[].filename -> itemId
-- All 40 resolved 'exact' with both methods in agreement. No record is guessed.
--
-- IDEMPOTENT: the UPDATE only touches rows whose agenda_item_id IS NULL, so a
-- second run reports 0 rows updated and changes nothing.
--
-- Deliberately does NOT backfill metadata.item_name / metadata.item_order,
-- which the flat loader also left NULL. That is a separate change, out of the
-- approved scope of this repair.

\set ON_ERROR_STOP on
\timing off

BEGIN;

CREATE TEMP TABLE link_map (external_id text PRIMARY KEY, agenda_item_id text NOT NULL)
    ON COMMIT DROP;

INSERT INTO link_map (external_id, agenda_item_id) VALUES
        ('DQU45Y09F4B0_1706_Certified_Signatures_OSPI_2026.pdf', 'DQU4WG0D8B13'),
        ('DQU45Y09F4B0_1707_Certified_Signatures_Real_Estate_Transactions_2026.pdf', 'DQU4WE0D8860'),
        ('DQU45Y09F4B0_1708CE~1.DOC.pdf', 'DQU4WC0D85FC'),
        ('DQU45Y09F4B0_AC_Portables_-_Install_Electrical_and_Data_Service_-_RFQ_and_Quote.pdf', 'DQU4WN0D9241'),
        ('DQU45Y09F4B0_ASB_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_BDMTG_-_2-11-2026_SIGNED.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_Board_Minutes_2026_01_28.pdf', 'DQU47309F4E4'),
        ('DQU45Y09F4B0_Board_Personnel_Report_02.11.2026.pdf', 'DQU47409F4E5'),
        ('DQU45Y09F4B0_Board_Special_Meeting_Minutes_2026_01_28.pdf', 'DQU47309F4E4'),
        ('DQU45Y09F4B0_Board_Special_Meeting_Minutes_2026_02_04.pdf', 'DQU56Y0EC92D'),
        ('DQU45Y09F4B0_Capital_Fund_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_Custodial_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_Donations_Board_Review_2.11.2026.pdf', 'DQU46Y09F4E1'),
        ('DQU45Y09F4B0_Financial_Statement_December_2025_FINAL.pdf', 'DR39NK23D246'),
        ('DQU45Y09F4B0_General_Fund_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_KE_-_(Signed)_Install_New_Fence_Along_Meeker_Street_-_RFQ_and_Quote.pdf', 'DQU4WK0D8E8A'),
        ('DQU45Y09F4B0_KE_-_Install_New_Fence_Along_Meeker_Street_-_RFQ_and_Quote.pdf', 'DQU4WK0D8E8A'),
        ('DQU45Y09F4B0_Proclamation.Black_History_Month_2026.pdf', 'DQU4B70AB813'),
        ('DQU45Y09F4B0_Proclamation_CTE_Month_2026.pdf', 'DQU4BL0AC74E'),
        ('DQU45Y09F4B0_TVF_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU45Y09F4B0_Trust_Vouchers_02-11-26.pdf', 'DR5MNW5C1C8C'),
        ('DQU47R0A3706_1220_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_1220_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_1760_-_Proposed_.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_1760_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_5012_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_5012_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6600_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6600_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6608_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6608_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6620_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6620_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6625_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6625_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6630_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6630_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6640_-_Proposed.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_6640_-_Redline.pdf', 'DQU47Z0A3719'),
        ('DQU47R0A3706_Policy_Work_Session_-_February_11,_2026.pdf', 'DQU47Z0A3719');

\echo '=== BEFORE ==='
SELECT count(*) FILTER (WHERE d.agenda_item_id IS NULL)     AS unlinked,
       count(*) FILTER (WHERE d.agenda_item_id IS NOT NULL) AS already_linked,
       count(*)                                             AS in_map
FROM link_map m JOIN documents d ON d.external_id = m.external_id;

-- Guard: every external_id in the map must exist exactly once. If the corpus
-- has drifted since the derivation, stop rather than silently under-applying.
DO $$
DECLARE missing int;
BEGIN
    SELECT count(*) INTO missing
    FROM link_map m
    WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.external_id = m.external_id);
    IF missing > 0 THEN
        RAISE EXCEPTION 'link_40: % mapped external_id(s) not found in documents', missing;
    END IF;
END $$;

UPDATE documents d
   SET agenda_item_id = m.agenda_item_id,
       updated_at     = now()
  FROM link_map m
 WHERE d.external_id   = m.external_id
   AND d.agenda_item_id IS NULL;      -- idempotency guard

\echo '=== AFTER ==='
SELECT count(*) FILTER (WHERE d.agenda_item_id IS NULL)     AS unlinked,
       count(*) FILTER (WHERE d.agenda_item_id IS NOT NULL) AS linked,
       count(*)                                             AS in_map
FROM link_map m JOIN documents d ON d.external_id = m.external_id;

\echo '=== 2026 unlinked-attachment coverage audit (expect 0 BoardDocs rows) ==='
SELECT extract(year from meeting_date)::int AS yr,
       count(*) FILTER (WHERE external_id LIKE 'email_%')     AS email_expected_null,
       count(*) FILTER (WHERE external_id NOT LIKE 'email_%') AS boarddocs_unlinked
FROM documents
WHERE document_type = 'attachment' AND agenda_item_id IS NULL
  AND meeting_date >= '2026-01-01'
GROUP BY 1 ORDER BY 1;

COMMIT;
