-- delete_31.sql — remove the 31 cross-layout duplicate attachment rows.
--
-- NOT EXECUTED. Written for operator review (Part 1); run only after approval.
-- Run AFTER link_40.sql: both read the same 2026 attachment rows.
--
-- Each row below was proven a true duplicate by
-- scripts/ingest-repair-2026-09-13/verify_duplicates.py, which compared the
-- SHA-256 of the actual file BYTES of the flat copy and the kept copy. Equal
-- file_size_bytes was NOT accepted as proof. All 31 pairs hashed identically
-- and all 31 kept rows carry an agenda_item_id.
--
-- Deletes the FLAT row and keeps the LINKED row. chunks and document_pages
-- follow via ON DELETE CASCADE. No facts.* row references any of the 31
-- (verified: all facts references point at the orphan set), so no non-cascading
-- foreign key can be violated -- but the guard below re-checks that at runtime.
--
-- IDEMPOTENT: the DELETE is keyed on rows that still exist and are still
-- unlinked, so a second run deletes 0 rows.
--
-- Qdrant is NOT touched. Point IDs for the deleted chunks are listed below and
-- in backups/ingest-repair-2026-09-13/qdrant_points_dup31.jsonl for a separate
-- cleanup task.

\set ON_ERROR_STOP on
\timing off

BEGIN;

CREATE TEMP TABLE dup_map (
    flat_external_id text PRIMARY KEY,
    kept_external_id text NOT NULL,
    sha256           text NOT NULL
) ON COMMIT DROP;

INSERT INTO dup_map (flat_external_id, kept_external_id, sha256) VALUES
        ('DQHKYP542B0C_1210_-_Proposed.pdf', 'DQHKYP542B0C_DQHLJD5650A5_1210 - Proposed.pdf', '2cd53ca1527fed69b865f9ea6b5470862c6113ccbd6bca57a66d05c034ff904f'),
        ('DQHKYP542B0C_1210_-_Redline.pdf', 'DQHKYP542B0C_DQHLJD5650A5_1210 - Redline.pdf', '7a6432c86a996ab5886c75463567304037518a3245832872af93829df18873cc'),
        ('DQHKYP542B0C_3410_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLHS5641D9_3410 - Proposed KSD.pdf', '50d456d4df3dbfe9aa00cb9592fd746dd3039ccfc0ca8437288ab4a45c520712'),
        ('DQHKYP542B0C_3410_-_Redline.pdf', 'DQHKYP542B0C_DQHLHS5641D9_3410 - Redline.pdf', '0dbaa75207a9f7132ba1057929950222ba38f06be2e70ac1c2f644d24a8bf2db'),
        ('DQHKYP542B0C_3412_-_Proposed.pdf', 'DQHKYP542B0C_DQHLHM563ED5_3412 - Proposed.pdf', '6c81d22675d758404b5c29390a227e10b130a3c932457ea33e9bb99a15a47a9f'),
        ('DQHKYP542B0C_3412_-_Redline.pdf', 'DQHKYP542B0C_DQHLHM563ED5_3412 - Redline.pdf', 'ca1f10008a67083e3e6aa2758355329b9a412b8edbaeac31141a1b13b2098d5e'),
        ('DQHKYP542B0C_3416_-_Proposed.pdf', 'DQHKYP542B0C_DQHLH75631B6_3416 - Proposed.pdf', '26e80b8c59b7fe8ed3f91ca94daf6b453fabffee5586c0b647cb6014f075db72'),
        ('DQHKYP542B0C_3416_-_Redline.pdf', 'DQHKYP542B0C_DQHLH75631B6_3416 - Redline.pdf', '09bc92485672c04cb78fe861be857fb67861a91e53a596f40c991314feec213b'),
        ('DQHKYP542B0C_3417_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLGY562CD4_3417 - Proposed KSD.pdf', '06080632e234e0b6fef9e53f5a7b98a8a905dc306a12fdb79a7a04d929eb6bbb'),
        ('DQHKYP542B0C_3417_-_Redline.pdf', 'DQHKYP542B0C_DQHLGY562CD4_3417 - Redline.pdf', 'ffa6833be8fbc05a44c623e9c9a67157004cedd6c699dcaa7c081651910073e5'),
        ('DQHKYP542B0C_3418_-_Proposed.pdf', 'DQHKYP542B0C_DQHLGV562A10_3418 - Proposed.pdf', 'ccd3fbf728f56906ba0a3f07188d8ff1d47defbf2056863cec0ad4749235b7ba'),
        ('DQHKYP542B0C_3418_-_Redline.pdf', 'DQHKYP542B0C_DQHLGV562A10_3418 - Redline.pdf', 'ace0470ab6760580e0cd8c6fb8b869a03632e0644eda46d8e2a874251240d154'),
        ('DQHKYP542B0C_3419_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLGH562490_3419 - Proposed KSD.pdf', 'bf37f77b6bec4ad6ecae084141c8d1975663f3b91d8397baebf925831ce14e2d'),
        ('DQHKYP542B0C_3419_-_Redline.pdf', 'DQHKYP542B0C_DQHLGH562490_3419 - Redline.pdf', '4ee519b982c2d39368546c0a3320a83a0e69cdee2b893c2394c5b79938a04a67'),
        ('DQHKYP542B0C_3420_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLGF562235_3420 - Proposed KSD.pdf', 'cd047bfb02a5da8bc1ac74afb2b9189018d6f1e78ae55f323b9233a2fcb2c2a6'),
        ('DQHKYP542B0C_3420_-_Redline.pdf', 'DQHKYP542B0C_DQHLGF562235_3420 - Redline.pdf', '75eb7133598b82f9bc8e38c5a8a1a08b1c8ba9bdb0ba0b7b7c145bc0dd544843'),
        ('DQHKYP542B0C_5000_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLGE561FC7_5000 - Proposed KSD.pdf', 'ed38191525b0f569e0192d00248e0daae3f4f6b97538f167d7bc4882df40aefe'),
        ('DQHKYP542B0C_5000_-_Redline.pdf', 'DQHKYP542B0C_DQHLGE561FC7_5000 - Redline.pdf', 'd8d4db1386c930acdb9127204ec62a77d797a9bf5aa7b62d7161f409ced6a13f'),
        ('DQHKYP542B0C_5010_-_Proposed_KSD.pdf', 'DQHKYP542B0C_DQHLGB561CDD_5010 - Proposed KSD.pdf', '7ac96530206ef3f0197069be36a5c87efe481db267058550bb49ca189071d2da'),
        ('DQHKYP542B0C_5010_-_Redline.pdf', 'DQHKYP542B0C_DQHLGB561CDD_5010 - Redline.pdf', 'ef8cb499fe7bace6f2cd36b5b47a86dadbd0f576b6fa37b640e4f8dc90364340'),
        ('DQHKYP542B0C_5281_-_Proposed.pdf', 'DQHKYP542B0C_DQHLGA561A25_5281 - Proposed.pdf', '779609e87c836f2c87361f248f202d8fe3a9a9372af48d799d536cb486ead87a'),
        ('DQHKYP542B0C_5281_-_Redline.pdf', 'DQHKYP542B0C_DQHLGA561A25_5281 - Redline.pdf', 'ecac34a85561b881044f9de96051b80bb6121e8ffa8456c054bae637b96a32df'),
        ('DQHKYP542B0C_5283_-_Proposed.pdf', 'DQHKYP542B0C_DQHLG75617A6_5283 - Proposed.pdf', '6888b1277e38838a3e06f3b471d11f25be6415bdd166b2890a1959078881dcf9'),
        ('DQHKYP542B0C_5th_Grade_Outdoor_Education_Survey_25-26_Analysis_pdf.pdf', 'DQHKYP542B0C_DQHL4R54C65D_5th Grade Outdoor Education Survey 25-26_Analysis pdf.pdf', 'd690e0f900534d0d309e67e6181b97654bbad1a82187ebfa91f8de625761cc2a'),
        ('DQHKYP542B0C_Board_Presentation_Winter_2026_.pdf', 'DQHKYP542B0C_DQHL4N54C391_Board Presentation Winter 2026_.pdf', '0614223d32d9dd5d5500c1c8254ebcafc9f1fc641ef7ebe3dfe5667786106c97'),
        ('DQHKYP542B0C_Donations_Board_Review_1.28.2026.pdf', 'DQHKYP542B0C_DQHKZM542B58_Donations Board Review 1.28.2026.pdf', 'a9251c55630e7240fcf427f41fde090a8ea73bec1a3d5e2c2e5a015633f4ca19'),
        ('DQHKYP542B0C_KSD_PowerPoint_Presentation_East_Hill_updated_pptx.pdf', 'DQHKYP542B0C_DQHKZF542B50_KSD PowerPoint Presentation East Hill updated pptx.pdf', 'ce1a22936494143ffbbda0034ea6cf7313808c41bf975a3f2cb20044e1bc3975'),
        ('DQHKYP542B0C_Meridian_Elementary_Student_Presentation.pdf', 'DQHKYP542B0C_DQHUAA7B15EB_Meridian Elementary Student Presentation.pdf', '09f1f5148cd73e83874d7ed44790c7d33620c6a3d1935932878fc99493415faf'),
        ('DQHKYP542B0C_Proclamation._National_School_Counseling_Week_2026.pdf', 'DQHKYP542B0C_DQM8BB1D7D26_Proclamation. National School Counseling Week 2026.pdf', 'd1d29bc6e54f26cd13ef01dda9eb5e67bc9b1e64b797fb5da2a1930f96085d96'),
        ('DQHKYP542B0C_State_of_Washington_Korean_American_Day_Proclamation.pdf', 'DQHKYP542B0C_DQKPXP66C892_State of Washington Korean American Day Proclamation.pdf', '7020a7391d893518502a9605e435ac6933f6756a4cfcab3ab80779eceb853c71'),
        ('DQU2PT0331CA_Budget_Update_2.4.26.pdf', 'DQU2PT0331CA_DQU2Q30331E4_Budget Update 2.4.26.pdf', '17c1fe182c0cf86090330c42b38a1ddcbe970f7098ed7f29acdabb047fdc8298');

\echo '=== BEFORE ==='
SELECT (SELECT count(*) FROM dup_map)                                             AS pairs_in_map,
       (SELECT count(*) FROM documents d JOIN dup_map m ON d.external_id = m.flat_external_id) AS flat_rows_present,
       (SELECT count(*) FROM documents d JOIN dup_map m ON d.external_id = m.kept_external_id) AS kept_rows_present,
       (SELECT count(*) FROM chunks c
          WHERE c.document_id IN (SELECT d.id FROM documents d
                                  JOIN dup_map m ON d.external_id = m.flat_external_id)) AS chunks_to_cascade;

\echo '=== Qdrant points that will be orphaned (for the SEPARATE cleanup task) ==='
SELECT c.qdrant_point_id, c.id AS chunk_id, d.external_id
FROM chunks c
JOIN documents d ON d.id = c.document_id
JOIN dup_map m   ON d.external_id = m.flat_external_id
WHERE c.qdrant_point_id IS NOT NULL
ORDER BY d.external_id, c.chunk_index;

-- Guard 1: every kept counterpart must still exist AND still be linked.
-- Never delete a row whose surviving twin has gone missing.
DO $$
DECLARE bad int;
BEGIN
    SELECT count(*) INTO bad
    FROM dup_map m
    WHERE NOT EXISTS (
        SELECT 1 FROM documents d
        WHERE d.external_id = m.kept_external_id AND d.agenda_item_id IS NOT NULL);
    IF bad > 0 THEN
        RAISE EXCEPTION 'delete_31: % kept counterpart(s) missing or unlinked -- aborting', bad;
    END IF;
END $$;

-- Guard 2: no facts.* row may reference a row we are about to delete.
DO $$
DECLARE bad int;
BEGIN
    SELECT count(*) INTO bad
    FROM documents d JOIN dup_map m ON d.external_id = m.flat_external_id
    WHERE d.id IN (SELECT minutes_document_id FROM facts.meeting
                   UNION ALL SELECT locator_document_id FROM facts.meeting
                   UNION ALL SELECT locator_document_id FROM facts.attendance
                   UNION ALL SELECT locator_document_id FROM facts.motion
                   UNION ALL SELECT locator_document_id FROM facts.vote
                   UNION ALL SELECT locator_document_id FROM facts.executive_session
                   UNION ALL SELECT document_id        FROM facts.minutes_parse_log);
    IF bad > 0 THEN
        RAISE EXCEPTION 'delete_31: % row(s) referenced by facts.* -- aborting', bad;
    END IF;
END $$;

DELETE FROM documents d
 USING dup_map m
 WHERE d.external_id    = m.flat_external_id
   AND d.agenda_item_id IS NULL;      -- idempotency + never delete a linked row

\echo '=== AFTER ==='
SELECT (SELECT count(*) FROM dup_map)                                             AS pairs_in_map,
       (SELECT count(*) FROM documents d JOIN dup_map m ON d.external_id = m.flat_external_id) AS flat_rows_remaining,
       (SELECT count(*) FROM documents d JOIN dup_map m ON d.external_id = m.kept_external_id) AS kept_rows_remaining,
       (SELECT count(*) FROM chunks c
          WHERE c.document_id IN (SELECT d.id FROM documents d
                                  JOIN dup_map m ON d.external_id = m.flat_external_id)) AS chunks_remaining;

\echo '=== every kept record still resolves ==='
SELECT count(*) AS kept_resolvable
FROM dup_map m JOIN documents d ON d.external_id = m.kept_external_id
WHERE d.agenda_item_id IS NOT NULL AND d.processing_status = 'complete';

COMMIT;
