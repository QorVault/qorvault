# BoardDocs OCR Service

FastAPI microservice for 3-tier PDF text extraction with GPU-accelerated OCR fallback.

## Extraction Tiers

1. **Digital** — `pypdf` extracts embedded text (fast, no GPU)
2. **Docling EasyOCR** — Docling with built-in EasyOCR for scanned pages
3. **Docling Surya** — Docling with Surya OCR for GPU-accelerated extraction

Falls through tiers automatically based on character count thresholds.

## Office File Support

Also handles `.docx`, `.pptx`, `.xlsx`, `.doc`, `.ppt`, `.xls` via python-docx, python-pptx, and openpyxl.

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

Requires Python 3.12 and AMD ROCm runtime. The setup script creates a virtualenv, installs PyTorch with ROCm 6.2 support, and verifies GPU access.

## Running

```bash
source .venv/bin/activate
uvicorn ocr_service.main:app --host 127.0.0.1 --port 8001
```

Or install the systemd service:

```bash
sudo cp ocr_service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ocr_service
```

## API Endpoints

- `POST /extract` — Extract text from a PDF or Office file
- `GET /health` — Service health and GPU status
- `GET /stats` — Extraction statistics

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
