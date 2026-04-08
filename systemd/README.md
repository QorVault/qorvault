# Systemd Service Files

This directory contains copies of all systemd user service files used by the BoardDocs RAG system. The actual installed copies live at `~/.config/systemd/user/`.

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| `boarddocs-postgres` | 5432 | PostgreSQL 16 + pgvector (Podman quadlet) |
| `boarddocs-qdrant` | 6333, 6334 | Qdrant v1.9.0 vector database (Podman quadlet) |
| `ksd-ocr-service` | 8001 | OCR text extraction (Docling + RapidOCR, CPU-only) |
| `ksd-llama-server` | 8080 | Local LLM inference (Qwen3-30B via llama.cpp) |
| `ksd-rag-api` | 8000 | RAG query API (FastAPI + Qdrant + Claude) |
| `container-open-webui` | 3000 | Web chat interface (Podman container) |

## Dependencies

```
boarddocs-postgres ──┐
                     ├──> ksd-rag-api ──┐
boarddocs-qdrant ────┘                  ├──> container-open-webui
                                        │
ksd-llama-server ───────────────────────┘

ksd-ocr-service (independent — only needed during document processing)
```

- **PostgreSQL and Qdrant** must be running before starting the RAG API.
- **ksd-rag-api** and **ksd-llama-server** must be running before Open WebUI can function.
- **ksd-ocr-service** is only needed during batch document processing, not for query serving.

## Correct Startup Order

1. PostgreSQL and Qdrant start automatically on boot via Podman quadlets
2. `ksd-llama-server` — start after boot (takes ~30s to load the 34 GB model)
3. `ksd-rag-api` — start after PostgreSQL and Qdrant are healthy
4. `container-open-webui` — start after llama-server and RAG API are ready
5. `ksd-ocr-service` — start only when processing new documents

## Installation on a Fresh Machine

### 1. Install Podman quadlets (database containers)

The quadlet files are in `infrastructure/quadlets/` and are installed by the infrastructure setup script:

```bash
cd ~/ksd-boarddocs-rag/infrastructure
./setup.sh
```

Or install manually:

```bash
mkdir -p ~/.config/containers/systemd/
cp infrastructure/quadlets/boarddocs-postgres.container ~/.config/containers/systemd/
cp infrastructure/quadlets/boarddocs-qdrant.container ~/.config/containers/systemd/
systemctl --user daemon-reload
```

### 2. Install application services

```bash
cp systemd/ksd-ocr-service.service ~/.config/systemd/user/
cp systemd/ksd-llama-server.service ~/.config/systemd/user/
cp systemd/ksd-rag-api.service ~/.config/systemd/user/
cp systemd/container-open-webui.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

### 3. Enable auto-start

```bash
# Database containers auto-start via quadlets (already enabled)
systemctl --user enable ksd-llama-server
systemctl --user enable ksd-rag-api
systemctl --user enable container-open-webui
# OCR service: enable only if processing documents
# systemctl --user enable ksd-ocr-service
```

### 4. Enable lingering (start services at boot, before login)

```bash
sudo loginctl enable-linger $USER
```

## Verification

After starting all services, verify each one:

```bash
# PostgreSQL (substitute your POSTGRES_USER / POSTGRES_DB from .env)
podman exec boarddocs-postgres pg_isready -U qorvault -d qorvault

# Qdrant
curl -s http://127.0.0.1:6333/healthz

# OCR Service (only if running)
curl -s http://localhost:8001/stats

# llama-server
curl -s http://127.0.0.1:8080/v1/models

# RAG API
curl -s http://127.0.0.1:8000/health

# Open WebUI
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# Should return 200
```

## GPU-Specific Notes

Some integrated GPU drivers crash when PyTorch loads BERT-family models or
when llama.cpp uses its default `--fit auto` model placement.  These
services use the safe defaults below; remove them only if you have
verified your GPU driver is stable for the relevant workloads.

- **ksd-llama-server**: Uses `--fit off` to bypass llama.cpp's auto-fit
  logic, which can crash on some GPUs during model load.
- **ksd-ocr-service**: Hides the GPU entirely
  (`CUDA_VISIBLE_DEVICES=`, `HIP_VISIBLE_DEVICES=`) so Docling and
  PyTorch fall back to CPU.
- **ksd-rag-api**: Uses ONNX Runtime CPU for embeddings (not PyTorch).

## File Locations

| File in this directory | Installed location |
|---|---|
| `ksd-ocr-service.service` | `~/.config/systemd/user/ksd-ocr-service.service` |
| `ksd-llama-server.service` | `~/.config/systemd/user/ksd-llama-server.service` |
| `ksd-rag-api.service` | `~/.config/systemd/user/ksd-rag-api.service` |
| `container-open-webui.service` | `~/.config/systemd/user/container-open-webui.service` |

The Podman quadlet files for PostgreSQL and Qdrant are at:
- `infrastructure/quadlets/boarddocs-postgres.container` → `~/.config/containers/systemd/`
- `infrastructure/quadlets/boarddocs-qdrant.container` → `~/.config/containers/systemd/`
