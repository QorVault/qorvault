#!/usr/bin/env bash
# BoardDocs RAG Infrastructure — End-to-End Setup
# Idempotent: safe to run multiple times without errors or duplication.
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$INFRA_DIR"

# ── 1. Create .env from example if it doesn't exist ─────────────────
if [[ ! -f .env ]]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
else
    echo "==> .env already exists, keeping it"
fi

# Source env vars
set -a
source .env
set +a

# Resolve DB/user from .env with deployment-default fallbacks so the rest
# of the script doesn't hardcode the name.
PG_DB="${POSTGRES_DB:-qorvault}"
PG_USER="${POSTGRES_USER:-qorvault}"

# ── 2. Install Python dependencies ──────────────────────────────────
echo "==> Installing Python dependencies"
pip install --quiet -r requirements.txt

# ── 3. Pull container images ────────────────────────────────────────
echo "==> Pulling container images (if not already cached)"
podman pull docker.io/pgvector/pgvector:pg16
podman pull docker.io/qdrant/qdrant:v1.9.0

# ── 4. Stop and remove existing containers (safe for first run) ─────
echo "==> Ensuring clean container state"
podman stop boarddocs-postgres 2>/dev/null || true
podman stop boarddocs-qdrant 2>/dev/null || true
podman rm boarddocs-postgres 2>/dev/null || true
podman rm boarddocs-qdrant 2>/dev/null || true

# ── 5. Create named volumes if they don't exist ────────────────────
echo "==> Ensuring named volumes exist"
podman volume exists boarddocs_pgdata 2>/dev/null || podman volume create boarddocs_pgdata
podman volume exists boarddocs_qdrant_storage 2>/dev/null || podman volume create boarddocs_qdrant_storage

# ── 6. Start containers with podman-compose ─────────────────────────
echo "==> Starting containers with podman-compose"
podman-compose -f podman-compose.yml up -d

# ── 7. Wait for PostgreSQL to accept SQL connections ────────────────
echo "==> Waiting for PostgreSQL to be ready"
for i in $(seq 1 60); do
    if podman exec boarddocs-postgres psql -U "$PG_USER" -d "$PG_DB" -c "SELECT 1" >/dev/null 2>&1; then
        echo "    PostgreSQL is accepting connections (attempt $i)"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "ERROR: PostgreSQL did not become ready in 60 attempts"
        exit 1
    fi
    sleep 2
done

# ── 8. Apply schema (idempotent — init.sql uses IF NOT EXISTS) ──────
echo "==> Applying database schema"
podman exec -i boarddocs-postgres psql -U "$PG_USER" -d "$PG_DB" < init.sql

# ── 9. Wait for Qdrant to be ready ─────────────────────────────────
echo "==> Waiting for Qdrant to be ready"
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
        echo "    Qdrant is ready (attempt $i)"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "ERROR: Qdrant did not become ready in 30 attempts"
        exit 1
    fi
    sleep 2
done

# ── 10. Create Qdrant collection (idempotent) ──────────────────────
echo "==> Creating Qdrant collection boarddocs_chunks (if not exists)"
COLLECTION_EXISTS=$(curl -sf http://localhost:6333/collections/boarddocs_chunks | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('yes' if data.get('status') == 'ok' else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")

if [[ "$COLLECTION_EXISTS" == "yes" ]]; then
    echo "    Collection already exists, skipping creation"
else
    echo "    Creating collection..."
    curl -sf -X PUT "http://localhost:6333/collections/boarddocs_chunks" \
        -H "Content-Type: application/json" \
        -d '{
            "vectors": {
                "size": 1024,
                "distance": "Cosine"
            }
        }' > /dev/null
    echo "    Collection created"
fi

# ── 11. Create Qdrant payload indexes (idempotent) ─────────────────
echo "==> Creating Qdrant payload indexes"
for field_spec in "tenant_id:keyword" "document_type:keyword" "contains_table:bool"; do
    field="${field_spec%%:*}"
    ftype="${field_spec##*:}"

    curl -sf -X PUT "http://localhost:6333/collections/boarddocs_chunks/index" \
        -H "Content-Type: application/json" \
        -d "{\"field_name\": \"$field\", \"field_schema\": \"$ftype\"}" > /dev/null
    echo "    Index on '$field' ($ftype) ensured"
done

# ── 12. Install quadlet files ──────────────────────────────────────
echo "==> Installing quadlet unit files"
bash "$INFRA_DIR/quadlets/install_quadlets.sh"

# ── 13. Run verification ───────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Running infrastructure verification"
echo "=========================================="
echo ""
python3 "$INFRA_DIR/verify_infrastructure.py"
