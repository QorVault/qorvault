#!/usr/bin/env bash
# Setup script for the embedding pipeline virtualenv (Python 3.12 + ONNX Runtime CPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="/usr/bin/python3.12"

echo "=== Embedding Pipeline Setup (ONNX Runtime CPU) ==="

# Check Python 3.12
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python 3.12 not found at $PYTHON"
    echo "Install it with: sudo dnf install python3.12"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtualenv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtualenv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtualenv..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

# Install project dependencies (ONNX Runtime CPU — no PyTorch needed)
echo "Installing project dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-asyncio

# Verify ONNX Runtime
echo ""
echo "=== ONNX Runtime Verification ==="
python -c "
import onnxruntime as ort
print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available providers: {ort.get_available_providers()}')
print(f'CPU provider: {\"CPUExecutionProvider\" in ort.get_available_providers()}')
print()
print('No PyTorch or GPU required — ONNX Runtime runs mxbai-embed-large-v1 on CPU.')
"

# Run tests
echo ""
echo "=== Running Tests ==="
python -m pytest tests/ -v

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Dry run:       python -m embedding_pipeline --dry-run"
echo "Full run:      python -m embedding_pipeline"
