#!/usr/bin/env bash
# export_backup.sh — export every row affected by the 2026 ingest repair.
#
# Writes JSONL (one JSON object per row, full row, no column omitted) plus a
# SHA-256 manifest and a row-count file. Read-only against the corpus: every
# statement is a SELECT.
#
# Usage: ./export_backup.sh <output-dir>

set -euo pipefail

OUT="${1:?usage: export_backup.sh <output-dir>}"
mkdir -p "$OUT"

PSQL=(podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs -P pager=off -At)

# Canonical target-set CTE, kept byte-identical to target_set.sql.
TARGET_CTE=$(cat <<'SQL'
WITH unlinked AS (
    SELECT id, meeting_id, meeting_date, title, external_id
    FROM documents
    WHERE document_type = 'attachment' AND agenda_item_id IS NULL
      AND meeting_date >= '2026-01-01' AND external_id NOT LIKE 'email_%'
),
linked AS (
    SELECT meeting_id, regexp_replace(title, '\s+', '_', 'g') AS santitle
    FROM documents
    WHERE document_type = 'attachment' AND agenda_item_id IS NOT NULL
      AND meeting_date >= '2026-01-01'
),
target AS (
    SELECT u.id,
           CASE WHEN EXISTS (SELECT 1 FROM linked l
                             WHERE l.meeting_id = u.meeting_id AND l.santitle = u.title)
                THEN 'dup31' ELSE 'orphan40' END AS bucket
    FROM unlinked u
    UNION ALL
    SELECT id, 'pending48' FROM documents
    WHERE document_type = 'agenda_item' AND processing_status = 'pending'
)
SQL
)

dump() {  # dump <filename> <sql-after-target-cte>
    local file="$1" sql="$2"
    printf '%s\n%s\n' "$TARGET_CTE" "$sql" \
        | "${PSQL[@]}" > "$OUT/$file"
    printf '  %-28s %s rows\n' "$file" "$(wc -l < "$OUT/$file")"
}

echo "Exporting to $OUT"

# The 119 documents, with their bucket recorded alongside the full row.
dump documents.jsonl \
  "SELECT json_build_object('bucket', t.bucket, 'row', to_jsonb(d))::text
     FROM documents d JOIN target t ON t.id = d.id
    ORDER BY t.bucket, d.external_id;"

# Every chunk belonging to those documents (ON DELETE CASCADE would remove these).
dump chunks.jsonl \
  "SELECT to_jsonb(c)::text FROM chunks c
    WHERE c.document_id IN (SELECT id FROM target) ORDER BY c.document_id, c.chunk_index;"

# Every page row belonging to those documents (also ON DELETE CASCADE).
dump document_pages.jsonl \
  "SELECT to_jsonb(p)::text FROM document_pages p
    WHERE p.document_id IN (SELECT id FROM target) ORDER BY p.document_id;"

# Dependent facts.* rows. These FKs are NOT ON DELETE CASCADE, so they are both
# restore material and the reason a delete could be refused.
dump facts_meeting.jsonl \
  "SELECT to_jsonb(m)::text FROM facts.meeting m
    WHERE m.minutes_document_id IN (SELECT id FROM target)
       OR m.locator_document_id IN (SELECT id FROM target);"

dump facts_attendance.jsonl \
  "SELECT to_jsonb(a)::text FROM facts.attendance a
    WHERE a.locator_document_id IN (SELECT id FROM target);"

dump facts_motion.jsonl \
  "SELECT to_jsonb(x)::text FROM facts.motion x
    WHERE x.locator_document_id IN (SELECT id FROM target);"

dump facts_vote.jsonl \
  "SELECT to_jsonb(v)::text FROM facts.vote v
    WHERE v.locator_document_id IN (SELECT id FROM target);"

dump facts_executive_session.jsonl \
  "SELECT to_jsonb(e)::text FROM facts.executive_session e
    WHERE e.locator_document_id IN (SELECT id FROM target);"

dump facts_minutes_parse_log.jsonl \
  "SELECT to_jsonb(l)::text FROM facts.minutes_parse_log l
    WHERE l.document_id IN (SELECT id FROM target);"

# Qdrant point IDs for chunks of the delete set — needed for the separate
# Qdrant cleanup, which this task explicitly does NOT perform.
dump qdrant_points_dup31.jsonl \
  "SELECT json_build_object('chunk_id', c.id, 'qdrant_point_id', c.qdrant_point_id,
                            'document_id', c.document_id, 'embedding_status', c.embedding_status)::text
     FROM chunks c JOIN target t ON t.id = c.document_id
    WHERE t.bucket = 'dup31' ORDER BY c.document_id, c.chunk_index;"

# Row counts, independently recomputed from the exported files.
{
    echo "# Row counts for ingest-repair export"
    echo "# generated from the exported files themselves, not from the queries"
    for f in "$OUT"/*.jsonl; do
        printf '%s\t%s\n' "$(wc -l < "$f")" "$(basename "$f")"
    done
} > "$OUT/row_counts.txt"

# SHA-256 manifest over every exported file.
( cd "$OUT" && sha256sum ./*.jsonl row_counts.txt > manifest.sha256 )

echo "Manifest:"
cat "$OUT/manifest.sha256"
