#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Document Processor Setup ==="

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Installing dependencies..."
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet pytest pytest-asyncio pytest-mock

echo ""
echo "Running tests..."
python -m pytest tests/ -v

echo ""
echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"
echo "Dry run:       python -m document_processor --dry-run"
echo "Full run:      python -m document_processor"
