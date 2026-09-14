-- process_48.sql — before/after counts for the Stage 2 run on the 48 agenda items.
--
-- NOT EXECUTED. Written for operator review (Part 1); run only after approval.
--
-- This file does NOT process anything: Stage 2 is document_processor, a Python
-- service. Run this with :phase set to 'before', then run process_48.py, then
-- run it again with :phase set to 'after' and compare. process_48.py invokes
-- both phases itself, so running this by hand is only needed for an independent
-- second opinion.
--
--   psql -v phase=before -f process_48.sql
--   psql -v phase=after  -f process_48.sql
--
-- Every statement here is a SELECT. Safe to run at any time, any number of
-- times; it is read-only and therefore trivially idempotent.

\set ON_ERROR_STOP on
\timing off

\echo '=== phase ==='
SELECT :'phase' AS phase, now()::timestamp(0) AS at;

-- 1. Status distribution for the target meetings. The 48 start as 'pending'
--    and should all read 'complete' afterwards, with no 'failed'/'deferred'.
\echo '=== agenda_item status, 2026-03-25 meetings ==='
SELECT meeting_id,
       count(*) FILTER (WHERE processing_status = 'pending')  AS pending,
       count(*) FILTER (WHERE processing_status = 'complete') AS complete,
       count(*) FILTER (WHERE processing_status = 'deferred') AS deferred,
       count(*) FILTER (WHERE processing_status = 'failed')   AS failed,
       count(*)                                               AS total
FROM documents
WHERE document_type = 'agenda_item'
  AND meeting_id IN ('DS4MSK5CA5C5', 'DS4MVC5CCBE6', 'DSCM9K5A287D')
GROUP BY 1 ORDER BY 1;

-- 2. Corpus-wide pending count. Confirms the 48 are the only pending agenda
--    items, so a type-scoped processor run cannot touch anything else.
\echo '=== corpus-wide pending by type (48 agenda_item expected before) ==='
SELECT document_type, count(*) AS pending
FROM documents WHERE processing_status = 'pending'
GROUP BY 1 ORDER BY 1;

-- 3. Chunk coverage per document. Every one of the 48 must have >= 1 chunk
--    after the run; zero-chunk documents mean extraction produced no text.
\echo '=== chunk coverage for the 48 ==='
SELECT count(*)                                  AS documents,
       count(*) FILTER (WHERE chunk_count > 0)   AS with_chunks,
       count(*) FILTER (WHERE chunk_count = 0)   AS without_chunks,
       coalesce(sum(chunk_count), 0)             AS total_chunks
FROM (
    SELECT d.id, (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
    FROM documents d
    WHERE d.document_type = 'agenda_item'
      AND d.meeting_id IN ('DS4MSK5CA5C5', 'DS4MVC5CCBE6', 'DSCM9K5A287D')
) s;

-- 4. Embedding coverage for those chunks. Stage 3 (embedding_pipeline) is a
--    separate run; before it, chunks are expected to be 'pending' here.
\echo '=== embedding status of the 48 items chunks ==='
SELECT coalesce(c.embedding_status, '(none)') AS embedding_status,
       count(*) AS chunks,
       count(c.qdrant_point_id) AS with_qdrant_point
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.document_type = 'agenda_item'
  AND d.meeting_id IN ('DS4MSK5CA5C5', 'DS4MVC5CCBE6', 'DSCM9K5A287D')
GROUP BY 1 ORDER BY 1;

-- 5. Blast-radius control. This fingerprint covers every document NOT in the
--    48 and must be byte-identical before and after. If it changes, the run
--    touched something it should not have.
\echo '=== fingerprint of all OTHER documents (must be unchanged) ==='
SELECT count(*) AS other_documents,
       md5(string_agg(d.id::text || ':' || d.processing_status || ':' ||
                      d.updated_at::text, '|' ORDER BY d.id)) AS fingerprint
FROM documents d
WHERE NOT (d.document_type = 'agenda_item'
           AND d.meeting_id IN ('DS4MSK5CA5C5', 'DS4MVC5CCBE6', 'DSCM9K5A287D'));

-- 6. Chunk-count fingerprint for all other documents, so a stray chunk insert
--    against an unrelated document is caught even if its row did not change.
\echo '=== chunk total for all OTHER documents (must be unchanged) ==='
SELECT count(*) AS other_chunks
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE NOT (d.document_type = 'agenda_item'
           AND d.meeting_id IN ('DS4MSK5CA5C5', 'DS4MVC5CCBE6', 'DSCM9K5A287D'));
