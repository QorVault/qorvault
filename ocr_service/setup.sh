#!/usr/bin/env bash
# Setup script for the OCR service virtualenv (Python 3.12 + ROCm PyTorch)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="/usr/bin/python3.12"
PYTORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"

echo "=== OCR Service Setup ==="

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

# Install project dependencies first (docling pulls CUDA torch as a dep)
echo "Installing project dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Install PyTorch ROCm AFTER requirements to overwrite CUDA version
echo "Installing PyTorch with ROCm 6.2 support (overwriting CUDA build)..."
pip install --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Verify GPU
echo ""
echo "=== GPU Verification ==="
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_memory / (1024**3):.1f} GB')
else:
    print('WARNING: No GPU detected. OCR will fall back to CPU.')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: uvicorn ocr_service.main:app --host 127.0.0.1 --port 8001"
