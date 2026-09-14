-- target_set.sql — canonical definition of the 119 records in scope for the
-- 2026 ingest-incident repair. Read-only; defines CTEs only, executes nothing.
--
-- Inlined by process_48.sql, link_40.sql and delete_31.sql so that all three
-- scripts, and the backup export, agree on exactly which rows they mean.
--
-- Buckets:
--   dup31     — flat-layout attachment rows that duplicate a correctly linked
--               structured row in the same meeting (delete candidates)
--   orphan40  — flat-layout attachment rows with no linked counterpart; the
--               only copy of that content in the corpus (link candidates)
--   pending48 — agenda items loaded 2026-03-25 that Stage 2 never processed
--
-- The dup/orphan split reproduces the flat scraper's own sanitization rule:
-- it replaced spaces with underscores in filenames, so a flat row duplicates a
-- structured row iff replace(linked.title,' ','_') = flat.title within the same
-- meeting. The comparison runs linked -> sanitized (never the reverse) because
-- sanitization is lossy: a filename containing a genuine underscore cannot be
-- reconstructed from the sanitized form.

WITH unlinked AS (
    SELECT id, meeting_id, meeting_date, title, external_id
    FROM documents
    WHERE document_type = 'attachment'
      AND agenda_item_id IS NULL
      AND meeting_date >= '2026-01-01'
      AND external_id NOT LIKE 'email_%'   -- email attachments correctly have no agenda item
),
linked AS (
    SELECT meeting_id,
           agenda_item_id,
           title,
           regexp_replace(title, '\s+', '_', 'g') AS santitle
    FROM documents
    WHERE document_type = 'attachment'
      AND agenda_item_id IS NOT NULL
      AND meeting_date >= '2026-01-01'
),
target AS (
    SELECT u.id,
           u.meeting_id,
           u.meeting_date,
           u.title,
           u.external_id,
           CASE
               WHEN EXISTS (SELECT 1 FROM linked l
                            WHERE l.meeting_id = u.meeting_id
                              AND l.santitle = u.title)
               THEN 'dup31'
               ELSE 'orphan40'
           END AS bucket
    FROM unlinked u
    UNION ALL
    SELECT id, meeting_id, meeting_date, title, external_id, 'pending48'
    FROM documents
    WHERE document_type = 'agenda_item'
      AND processing_status = 'pending'
)
SELECT * FROM target;
