#!/usr/bin/env bash
# verify_restore.sh — prove the ingest-repair export actually restores.
#
# Loads the exported JSONL into a throwaway schema, compares row counts and a
# content-level checksum against the live tables, then drops ONLY that schema.
# Touches nothing in public.* or facts.* — every statement against the real
# corpus is a SELECT.
#
# Usage: ./verify_restore.sh <backup-dir>

set -euo pipefail

DIR="${1:?usage: verify_restore.sh <backup-dir>}"
SCRATCH="scratch_ingest_repair"

PSQL=(podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs -P pager=off)

echo "=== Creating scratch schema: $SCRATCH ==="
"${PSQL[@]}" -q <<SQL
DROP SCHEMA IF EXISTS ${SCRATCH} CASCADE;
CREATE SCHEMA ${SCRATCH};
-- Structure only: LIKE copies columns and types but no FKs, so the scratch
-- copy cannot reference (or endanger) anything in public.* or facts.*.
CREATE TABLE ${SCRATCH}.documents      (LIKE public.documents);
CREATE TABLE ${SCRATCH}.chunks         (LIKE public.chunks);
CREATE TABLE ${SCRATCH}.staging_doc    (j jsonb);
CREATE TABLE ${SCRATCH}.staging_chunk  (j jsonb);
SQL

echo "=== Loading exported JSONL into scratch ==="
# \copy reads from the client side; feeding via stdin keeps the files on the host.
#
# FORMAT csv, not the default text format. COPY's text format treats backslash
# sequences as escapes, so a literal \n inside a JSON string value would be
# turned back into a real newline and break the JSON. CSV format does no
# backslash processing. The delimiter and quote are set to control characters
# that cannot occur in the JSON output, so each line loads verbatim.
COPYOPTS="WITH (FORMAT csv, DELIMITER E'\x01', QUOTE E'\x02')"
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs -q \
    -c "\copy ${SCRATCH}.staging_doc (j) FROM STDIN ${COPYOPTS}" < "$DIR/documents.jsonl"
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs -q \
    -c "\copy ${SCRATCH}.staging_chunk (j) FROM STDIN ${COPYOPTS}" < "$DIR/chunks.jsonl"

"${PSQL[@]}" -q <<SQL
-- documents.jsonl wraps each row as {"bucket": ..., "row": {...}}
INSERT INTO ${SCRATCH}.documents
SELECT (jsonb_populate_record(NULL::public.documents, j->'row')).*
FROM ${SCRATCH}.staging_doc;

INSERT INTO ${SCRATCH}.chunks
SELECT (jsonb_populate_record(NULL::public.chunks, j)).*
FROM ${SCRATCH}.staging_chunk;
SQL

echo
echo "=== Row counts: restored vs live ==="
"${PSQL[@]}" <<SQL
WITH unlinked AS (
    SELECT id, meeting_id, title FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NULL
      AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%'),
linked AS (
    SELECT meeting_id, regexp_replace(title,'\s+','_','g') AS santitle FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
      AND meeting_date>='2026-01-01'),
target AS (
    SELECT u.id FROM unlinked u
    UNION ALL
    SELECT id FROM documents WHERE document_type='agenda_item' AND processing_status='pending')
SELECT 'documents' AS tbl,
       (SELECT count(*) FROM ${SCRATCH}.documents) AS restored,
       (SELECT count(*) FROM target)               AS live,
       CASE WHEN (SELECT count(*) FROM ${SCRATCH}.documents)
               = (SELECT count(*) FROM target) THEN 'MATCH' ELSE 'MISMATCH' END AS verdict
UNION ALL
SELECT 'chunks',
       (SELECT count(*) FROM ${SCRATCH}.chunks),
       (SELECT count(*) FROM chunks WHERE document_id IN (SELECT id FROM target)),
       CASE WHEN (SELECT count(*) FROM ${SCRATCH}.chunks)
               = (SELECT count(*) FROM chunks WHERE document_id IN (SELECT id FROM target))
            THEN 'MATCH' ELSE 'MISMATCH' END;
SQL

echo
echo "=== Content checksum: restored vs live (not just counts) ==="
# md5 over the ordered, fully-serialized rows. Equality here means every column
# of every row round-tripped through the export intact.
"${PSQL[@]}" <<SQL
WITH unlinked AS (
    SELECT id, meeting_id, title FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NULL
      AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%'),
target AS (
    SELECT u.id FROM unlinked u
    UNION ALL
    SELECT id FROM documents WHERE document_type='agenda_item' AND processing_status='pending')
SELECT 'documents' AS tbl,
       (SELECT md5(string_agg(t::text, '|' ORDER BY t.id))
          FROM ${SCRATCH}.documents t) AS restored_md5,
       (SELECT md5(string_agg(d::text, '|' ORDER BY d.id))
          FROM documents d WHERE d.id IN (SELECT id FROM target)) AS live_md5
UNION ALL
SELECT 'chunks',
       (SELECT md5(string_agg(t::text, '|' ORDER BY t.id)) FROM ${SCRATCH}.chunks t),
       (SELECT md5(string_agg(c::text, '|' ORDER BY c.id))
          FROM chunks c WHERE c.document_id IN (SELECT id FROM target));
SQL

echo
echo "=== Dropping scratch schema (and nothing else) ==="
"${PSQL[@]}" -c "DROP SCHEMA ${SCRATCH} CASCADE;"
"${PSQL[@]}" -c "SELECT count(*) AS scratch_schemas_remaining FROM information_schema.schemata WHERE schema_name = '${SCRATCH}';"
