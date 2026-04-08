#!/usr/bin/env bash
# Install Podman quadlet files for boarddocs containers.
# Idempotent: safe to run multiple times.
set -euo pipefail

QUADLET_DIR="$HOME/.config/containers/systemd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing quadlet files to $QUADLET_DIR"
mkdir -p "$QUADLET_DIR"

cp "$SCRIPT_DIR/boarddocs-postgres.container" "$QUADLET_DIR/"
cp "$SCRIPT_DIR/boarddocs-qdrant.container" "$QUADLET_DIR/"

echo "==> Reloading systemd user daemon"
systemctl --user daemon-reload

# Quadlet-generated units are auto-enabled via WantedBy=default.target.
# No need for 'systemctl enable'. Verify they are loaded:
echo "==> Verifying quadlet units are loaded"
systemctl --user list-units 'boarddocs*' --all --no-pager --no-legend | while read -r line; do
    echo "    $line"
done

echo "==> Quadlet installation complete"
echo "    Units will auto-start on next login (or reboot)."
echo "    To start them now via systemd: systemctl --user start boarddocs-postgres boarddocs-qdrant"
