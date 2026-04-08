#!/usr/bin/env bash
# Filesystem watcher: monitors project directory for code changes.
# Outputs JSON lines to logs/filesystem_events.jsonl.
# Runs as systemd user service ksd-fs-watcher.service.

set -uo pipefail

PROJECT_DIR="/home/qorvault/projects/ksd-boarddocs-rag"
OUTPUT_FILE="${PROJECT_DIR}/logs/filesystem_events.jsonl"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Map inotifywait events to simpler names
map_event() {
    case "$1" in
        CREATE*)   echo "create" ;;
        MODIFY*)   echo "modify" ;;
        DELETE*)   echo "delete" ;;
        MOVED_FROM*) echo "move" ;;
        MOVED_TO*)   echo "move" ;;
        *)         echo "other" ;;
    esac
}

# Get file extension
get_ext() {
    local f="$1"
    if [[ "$f" == *.* ]]; then
        echo ".${f##*.}"
    else
        echo ""
    fi
}

inotifywait -m -r \
    --exclude '(\.git/|__pycache__/|node_modules/|data/|input/|output/|\.swp$|\.tmp$|~$|\.jsonl$|\.log$|aggregator_cursor|\.active_session)' \
    -e create -e modify -e delete -e move \
    --format '%T %e %w%f' \
    --timefmt '%Y-%m-%dT%H:%M:%S' \
    "$PROJECT_DIR" 2>/dev/null | while IFS= read -r line; do
    # Parse: TIMESTAMP EVENTS FILEPATH
    ts=$(echo "$line" | awk '{print $1}')
    events=$(echo "$line" | awk '{print $2}')
    filepath=$(echo "$line" | awk '{$1=""; $2=""; print}' | sed 's/^  //')

    event_type=$(map_event "$events")
    ext=$(get_ext "$filepath")

    # Output JSON line
    printf '{"timestamp":"%sZ","event_type":"%s","file_path":"%s","file_extension":"%s","source":"inotifywait"}\n' \
        "$ts" "$event_type" "$filepath" "$ext" >> "$OUTPUT_FILE"
done
