#!/usr/bin/env bash
# Git auto-snapshot: commits all changes every minute when changes exist.
# Intended to run via cron: * * * * * /home/qorvault/projects/ksd-boarddocs-rag/scripts/git_auto_snapshot.sh
# Uses flock to prevent overlapping runs.

set -euo pipefail

PROJECT_DIR="/home/qorvault/projects/ksd-boarddocs-rag"
LOCK_FILE="/tmp/ksd_git_snapshot.lock"

# Acquire lock (non-blocking) — if another snapshot is running, exit silently
exec 200>"$LOCK_FILE"
flock -n 200 || exit 0

cd "$PROJECT_DIR"

# Ensure git repo exists
if [ ! -d .git ]; then
    git init
fi

# Check for any changes (staged, unstaged, or untracked)
if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    # No changes at all
    exit 0
fi

# Stage everything (respects .gitignore)
git add -A

# Count changed files
CHANGED=$(git diff --cached --name-only | wc -l)

# If nothing staged after add (e.g., only ignored files changed), exit
if [ "$CHANGED" -eq 0 ]; then
    exit 0
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
git commit -m "auto-snapshot ${TIMESTAMP} [files changed: ${CHANGED}]" --no-gpg-sign >/dev/null 2>&1 || true
