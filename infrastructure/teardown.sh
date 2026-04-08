#!/usr/bin/env bash
# Teardown: stops and removes boarddocs containers but PRESERVES data volumes.
# To also delete volumes, pass --volumes.
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Stopping boarddocs containers"
podman stop boarddocs-postgres boarddocs-qdrant 2>/dev/null || true

echo "==> Removing boarddocs containers"
podman rm boarddocs-postgres boarddocs-qdrant 2>/dev/null || true

if [[ "${1:-}" == "--volumes" ]]; then
    echo "==> Removing data volumes (ALL DATA WILL BE LOST)"
    podman volume rm boarddocs_pgdata boarddocs_qdrant_storage 2>/dev/null || true
    echo "==> Volumes removed"
else
    echo "==> Data volumes preserved (pass --volumes to remove them)"
fi

echo "==> Teardown complete"
