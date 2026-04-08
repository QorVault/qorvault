#!/usr/bin/env bash
# =============================================================================
# BoardDocs RAG — Nightly Database Backup Script
# =============================================================================
# Backs up PostgreSQL (pg_dump) and Qdrant (snapshot API) to a timestamped
# directory under ~/ksd-boarddocs-rag/backups/. Writes a manifest.json with
# record counts and file sizes. Deletes backups older than 7 days.
#
# Usage:
#   ./scripts/backup_databases.sh          # Run manually
#   crontab: 0 2 * * * /home/qorvault/projects/ksd-boarddocs-rag/scripts/backup_databases.sh
#
# Exit codes:
#   0 — all backups completed successfully
#   1 — one or more backups failed
# =============================================================================
set -euo pipefail

PROJECT_DIR="/home/qorvault/projects/ksd-boarddocs-rag"
BACKUP_ROOT="${PROJECT_DIR}/backups"
TODAY="$(date +%Y-%m-%d)"
BACKUP_DIR="${BACKUP_ROOT}/${TODAY}"
LOG_FILE="${BACKUP_ROOT}/backup.log"
QDRANT_URL="http://127.0.0.1:6333"

# Load .env so POSTGRES_DB / POSTGRES_USER are available, then resolve
# with deployment-default fallbacks.
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi
PG_DB="${POSTGRES_DB:-qorvault}"
PG_USER="${POSTGRES_USER:-qorvault}"

# Ensure directories exist
mkdir -p "${BACKUP_DIR}"
mkdir -p "${BACKUP_ROOT}"

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Backup started ==="

# Track overall success
FAILED=0

# ---------------------------------------------------------------------------
# 1. PostgreSQL backup via pg_dump inside the container
# ---------------------------------------------------------------------------
PG_DUMP_FILE="${BACKUP_DIR}/postgres_${PG_DB}.sql"

log "Dumping PostgreSQL database..."
if podman exec boarddocs-postgres pg_dump -U "$PG_USER" "$PG_DB" > "${PG_DUMP_FILE}" 2>>"${LOG_FILE}"; then
    # Verify dump is non-empty
    if [[ -s "${PG_DUMP_FILE}" ]]; then
        PG_SIZE=$(stat --format=%s "${PG_DUMP_FILE}")
        log "PostgreSQL dump complete: ${PG_DUMP_FILE} (${PG_SIZE} bytes)"
    else
        log "ERROR: PostgreSQL dump file is empty!"
        FAILED=1
    fi
else
    log "ERROR: pg_dump command failed!"
    FAILED=1
fi

if [[ $FAILED -eq 1 ]]; then
    log "=== Backup FAILED (PostgreSQL) ==="
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Qdrant snapshot via REST API
# ---------------------------------------------------------------------------
QDRANT_SNAPSHOT_FILE="${BACKUP_DIR}/qdrant_boarddocs_chunks.snapshot"

log "Creating Qdrant snapshot..."
SNAPSHOT_RESPONSE=$(curl -s -X POST "${QDRANT_URL}/collections/boarddocs_chunks/snapshots" 2>>"${LOG_FILE}")

if [[ -z "${SNAPSHOT_RESPONSE}" ]]; then
    log "ERROR: Qdrant snapshot API returned empty response!"
    FAILED=1
else
    # Parse the snapshot filename from the API response
    SNAPSHOT_NAME=$(echo "${SNAPSHOT_RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('status') == 'ok':
        print(data['result']['name'])
    else:
        print('')
except:
    print('')
" 2>/dev/null)

    if [[ -z "${SNAPSHOT_NAME}" ]]; then
        log "ERROR: Failed to parse Qdrant snapshot response: ${SNAPSHOT_RESPONSE}"
        FAILED=1
    else
        log "Qdrant snapshot created: ${SNAPSHOT_NAME}"

        # Find the snapshot file in the Qdrant volume
        # Check both possible volume names (quadlet vs compose naming)
        SNAPSHOT_SOURCE=""
        for VOLUME_NAME in boarddocs_qdrant_storage infrastructure_boarddocs_qdrant_storage; do
            MOUNTPOINT=$(podman volume inspect "${VOLUME_NAME}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data[0]['Mountpoint'])
except:
    print('')
" 2>/dev/null)
            if [[ -n "${MOUNTPOINT}" ]]; then
                CANDIDATE="${MOUNTPOINT}/collections/boarddocs_chunks/snapshots/${SNAPSHOT_NAME}"
                if [[ -f "${CANDIDATE}" ]]; then
                    SNAPSHOT_SOURCE="${CANDIDATE}"
                    break
                fi
            fi
        done

        if [[ -n "${SNAPSHOT_SOURCE}" ]]; then
            cp "${SNAPSHOT_SOURCE}" "${QDRANT_SNAPSHOT_FILE}"
            QDRANT_SIZE=$(stat --format=%s "${QDRANT_SNAPSHOT_FILE}")
            log "Qdrant snapshot copied: ${QDRANT_SNAPSHOT_FILE} (${QDRANT_SIZE} bytes)"
        else
            # Fallback: download via the Qdrant HTTP API
            log "Snapshot file not found on disk, downloading via API..."
            if curl -s -o "${QDRANT_SNAPSHOT_FILE}" "${QDRANT_URL}/collections/boarddocs_chunks/snapshots/${SNAPSHOT_NAME}" 2>>"${LOG_FILE}"; then
                if [[ -s "${QDRANT_SNAPSHOT_FILE}" ]]; then
                    QDRANT_SIZE=$(stat --format=%s "${QDRANT_SNAPSHOT_FILE}")
                    log "Qdrant snapshot downloaded: ${QDRANT_SNAPSHOT_FILE} (${QDRANT_SIZE} bytes)"
                else
                    log "ERROR: Downloaded Qdrant snapshot is empty!"
                    FAILED=1
                fi
            else
                log "ERROR: Failed to download Qdrant snapshot!"
                FAILED=1
            fi
        fi
    fi
fi

if [[ $FAILED -eq 1 ]]; then
    log "=== Backup FAILED (Qdrant) ==="
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Gather counts for the manifest
# ---------------------------------------------------------------------------
log "Gathering record counts..."

# PostgreSQL counts
DOC_COUNT=$(podman exec boarddocs-postgres psql -U "$PG_USER" -d "$PG_DB" -t -A \
    -c "SELECT COUNT(*) FROM documents WHERE tenant_id='kent_sd'" 2>/dev/null | tr -d '[:space:]')
CHUNK_COUNT=$(podman exec boarddocs-postgres psql -U "$PG_USER" -d "$PG_DB" -t -A \
    -c "SELECT COUNT(*) FROM chunks WHERE tenant_id='kent_sd'" 2>/dev/null | tr -d '[:space:]')

# Qdrant vector count
QDRANT_INFO=$(curl -s "${QDRANT_URL}/collections/boarddocs_chunks" 2>/dev/null)
VECTOR_COUNT=$(echo "${QDRANT_INFO}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['result']['points_count'])
except:
    print(0)
" 2>/dev/null)

log "Counts — documents: ${DOC_COUNT}, chunks: ${CHUNK_COUNT}, vectors: ${VECTOR_COUNT}"

# ---------------------------------------------------------------------------
# 4. Write manifest.json
# ---------------------------------------------------------------------------
PG_SIZE=${PG_SIZE:-0}
QDRANT_SIZE=${QDRANT_SIZE:-0}

python3 -c "
import json
from datetime import datetime, timezone

manifest = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'postgres_dump_size_bytes': ${PG_SIZE},
    'qdrant_snapshot_size_bytes': ${QDRANT_SIZE},
    'postgres_document_count': ${DOC_COUNT:-0},
    'postgres_chunk_count': ${CHUNK_COUNT:-0},
    'qdrant_vector_count': ${VECTOR_COUNT:-0},
}

with open('${BACKUP_DIR}/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
" 2>>"${LOG_FILE}"

log "Manifest written: ${BACKUP_DIR}/manifest.json"

# ---------------------------------------------------------------------------
# 5. Delete backups older than 7 days
# ---------------------------------------------------------------------------
log "Cleaning up old backups..."
DELETED_COUNT=0
while IFS= read -r old_dir; do
    log "  Deleting old backup: ${old_dir}"
    rm -rf "${old_dir}"
    DELETED_COUNT=$((DELETED_COUNT + 1))
done < <(find "${BACKUP_ROOT}" -maxdepth 1 -type d -mtime +7 -not -path "${BACKUP_ROOT}" 2>/dev/null)

if [[ $DELETED_COUNT -gt 0 ]]; then
    log "Deleted ${DELETED_COUNT} old backup(s)"
else
    log "No old backups to clean up"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "=== Backup completed successfully ==="
log "  PostgreSQL: ${PG_SIZE} bytes"
log "  Qdrant: ${QDRANT_SIZE} bytes"
log "  Documents: ${DOC_COUNT}, Chunks: ${CHUNK_COUNT}, Vectors: ${VECTOR_COUNT}"
log ""

exit 0
