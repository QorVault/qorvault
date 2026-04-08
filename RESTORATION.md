# System Restoration Guide

How to restore the entire BoardDocs RAG system from scratch onto a fresh Fedora 43 machine. Written for someone with Linux experience but no prior knowledge of this project.

## 1. Prerequisites

### Hardware requirements

- **Minimum**: Any x86_64 machine with enough RAM to hold the embedding
  model (CPU-only, no local LLM).  Several GB of free RAM and a few CPU
  cores are sufficient for query-time RAG.
- **Recommended**: A self-hosted server with sufficient unified RAM for the
  embedding model and local LLM inference.  Pick a machine sized for the
  largest local model you plan to run.
- **GPU (optional)**: Any GPU with a stable inference runtime (CUDA, ROCm,
  or Vulkan) if you want to run llama-server locally.  CPU-only is fine
  if you use a remote LLM API instead.

### Software to install

```bash
# System packages
sudo dnf install -y podman podman-compose git python3 python3-pip curl jq

# GPU runtime (only needed for llama-server GPU inference)
# Follow your vendor's official install guide:
#   - AMD ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#   - NVIDIA CUDA: https://developer.nvidia.com/cuda-downloads

# llama.cpp (only needed for local LLM)
# Build from source: https://github.com/ggerganov/llama.cpp
# Install the resulting llama-server binary somewhere on PATH (e.g.
# /opt/llama.cpp/build/bin/llama-server) and update the systemd unit
# in systemd/ksd-llama-server.service to point at it.

# Enable lingering so user services start at boot
sudo loginctl enable-linger $USER
```

### Verify prerequisites

```bash
podman --version          # 4.0+ required
podman-compose version    # 1.0+ required
python3 --version         # 3.11+ required
git --version
```

## 2. Clone the Repository

```bash
cd ~
git clone https://github.com/dcafter/ksd-boarddocs-rag.git
cd ksd-boarddocs-rag
```

## 3. Infrastructure Containers (PostgreSQL + Qdrant)

The infrastructure setup script handles everything: pulling images, starting containers, applying the database schema, creating the Qdrant collection, and installing Podman quadlets for auto-start.

```bash
cd ~/ksd-boarddocs-rag/infrastructure

# Create the environment file (edit passwords if desired)
cp .env.example .env

# Run the full setup (idempotent — safe to run multiple times)
chmod +x setup.sh teardown.sh quadlets/install_quadlets.sh
./setup.sh
```

This will:
1. Pull `pgvector/pgvector:pg16` and `qdrant/qdrant:v1.9.0` container images
2. Start both containers via podman-compose
3. Apply the database schema (tables, indexes, extensions)
4. Create the `boarddocs_chunks` Qdrant collection (1024 dimensions, Cosine)
5. Install Podman quadlet files so containers auto-start on login
6. Run 10-point verification checks

### Verify

```bash
# PostgreSQL (substitute your POSTGRES_USER / POSTGRES_DB from .env)
podman exec boarddocs-postgres pg_isready -U qorvault -d qorvault

# Qdrant
curl -s http://127.0.0.1:6333/healthz
```

**Important**: Always use `127.0.0.1`, never `localhost`, for connecting to Podman containers. Fedora resolves localhost to IPv6 (::1) first, but Podman only binds IPv4.

## 4. Environment Configuration

```bash
cd ~/ksd-boarddocs-rag
cp .env.example .env
```

Edit `.env` and fill in at minimum:

| Variable | Required for | Where to get it |
|----------|-------------|-----------------|
| `ANTHROPIC_API_KEY` | RAG API query answering | https://console.anthropic.com/api-keys |
| `R2_ACCOUNT_ID` | Audio transcription uploads | Cloudflare R2 dashboard |
| `R2_ACCESS_KEY_ID` | Audio transcription uploads | Cloudflare R2 API tokens |
| `R2_SECRET_ACCESS_KEY` | Audio transcription uploads | Cloudflare R2 API tokens |
| `R2_BUCKET_NAME` | Audio transcription uploads | Cloudflare R2 dashboard |
| `RUNPOD_API_KEY` | GPU pod provisioning | https://www.runpod.io/console/user/settings |
| `HF_TOKEN` | Pyannote diarization models | https://huggingface.co/settings/tokens |

The `ANTHROPIC_API_KEY` is the only one needed for basic RAG query functionality. The R2, RunPod, and HF variables are only needed for the transcription pipeline.

## 5. Python Virtual Environments

Each component has its own virtualenv and requirements.txt. Create them in order:

### OCR Service

```bash
cd ~/ksd-boarddocs-rag/ocr_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

### Document Processor

```bash
cd ~/ksd-boarddocs-rag/document_processor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### Embedding Pipeline

```bash
cd ~/ksd-boarddocs-rag/embedding_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### RAG API

```bash
cd ~/ksd-boarddocs-rag/rag_api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### Note on venv paths

- OCR service uses `.venv` (dot prefix)
- All other components use `venv` (no dot)
- This matches the systemd service file paths

## 6. Systemd Services

### Install service files

```bash
# Application services
cp ~/ksd-boarddocs-rag/systemd/ksd-ocr-service.service ~/.config/systemd/user/
cp ~/ksd-boarddocs-rag/systemd/ksd-llama-server.service ~/.config/systemd/user/
cp ~/ksd-boarddocs-rag/systemd/ksd-rag-api.service ~/.config/systemd/user/
cp ~/ksd-boarddocs-rag/systemd/container-open-webui.service ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload
```

### Enable auto-start

```bash
systemctl --user enable ksd-llama-server
systemctl --user enable ksd-rag-api
systemctl --user enable container-open-webui
# OCR service: only enable if actively processing documents
# systemctl --user enable ksd-ocr-service
```

### Start services (in dependency order)

```bash
# 1. Database containers are already running via quadlets
# 2. LLM server (takes ~30s to load the 34 GB model)
systemctl --user start ksd-llama-server

# 3. RAG API (needs PostgreSQL + Qdrant + ANTHROPIC_API_KEY)
systemctl --user start ksd-rag-api

# 4. Open WebUI
systemctl --user start container-open-webui
```

See `systemd/README.md` for hardware-specific notes and dependency details.

## 7. Models

### Embedding model (required for RAG)

The ONNX embedding model is downloaded automatically on first run of the embedding pipeline or RAG API. It caches to:

```
embedding_pipeline/model_cache/mxbai-embed-large-v1-onnx/  (~1.3 GB)
```

No manual download needed.

### Chat model (required for Open WebUI local LLM)

Download the Qwen3-30B-A3B GGUF model (~34 GB):

```bash
sudo mkdir -p /opt/models/chat/qwen3-30b-q8_k_xl
# Download from Hugging Face (search for "Qwen3-30B-A3B-UD-Q8_K_XL.gguf")
# Place at: /opt/models/chat/qwen3-30b-q8_k_xl/Qwen3-30B-A3B-UD-Q8_K_XL.gguf
```

The ksd-llama-server service file expects this exact path. If you place the model elsewhere, update the `--model` path in `systemd/ksd-llama-server.service`.

### Embedding GGUF (optional, not currently used)

There is also `mxbai-embed-large-v1-f16.gguf` (~639 MB) at `/opt/models/embed/` but the system uses the ONNX format, not GGUF, for embeddings.

## 8. Database Restoration from Backup

If restoring from a backup (created by `scripts/backup_databases.sh`):

### Restore PostgreSQL

```bash
# The backup file is a plain SQL dump.  Substitute your POSTGRES_USER /
# POSTGRES_DB from .env if you overrode the defaults.
podman exec -i boarddocs-postgres psql -U qorvault -d qorvault < backups/YYYY-MM-DD/postgres_qorvault.sql
```

### Restore Qdrant

```bash
# Upload the snapshot to Qdrant via its REST API
# First, delete the existing collection if it exists
curl -X DELETE http://127.0.0.1:6333/collections/boarddocs_chunks

# Restore from snapshot file
curl -X POST "http://127.0.0.1:6333/collections/boarddocs_chunks/snapshots/upload" \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@backups/YYYY-MM-DD/qdrant_boarddocs_chunks.snapshot"
```

### Verify the restore

Check manifest.json from the backup to know expected counts:

```bash
cat backups/YYYY-MM-DD/manifest.json
```

Then verify:

```bash
# PostgreSQL document count
podman exec boarddocs-postgres psql -U qorvault -d qorvault -t -A \
    -c "SELECT COUNT(*) FROM documents WHERE tenant_id='kent_sd'"

# Qdrant vector count
curl -s http://127.0.0.1:6333/collections/boarddocs_chunks | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Vectors: {data['result']['points_count']}\")
"
```

## 9. Verification

Run these checks after restoration to verify everything is working:

```bash
# 1. PostgreSQL is accepting connections
podman exec boarddocs-postgres pg_isready -U qorvault -d qorvault
# Expected: accepting connections

# 2. Qdrant is healthy
curl -s http://127.0.0.1:6333/healthz
# Expected: {"title":"qdrant - vectorass engine","version":"..."}

# 3. Database has data
podman exec boarddocs-postgres psql -U qorvault -d qorvault -t -A \
    -c "SELECT COUNT(*) FROM documents WHERE tenant_id='kent_sd'"
# Expected: ~19660

# 4. Qdrant has vectors
curl -s http://127.0.0.1:6333/collections/boarddocs_chunks | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f\"Points: {d['result']['points_count']}, Status: {d['result']['status']}\")"
# Expected: Points: ~31000+, Status: green

# 5. RAG API is healthy
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
# Expected: {"status": "ok"}

# 6. RAG API health with details
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
# Expected: status=healthy, database=true, qdrant=true, embedder=true

# 7. llama-server is serving the model
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
# Expected: model list including Qwen3-30B

# 8. Open WebUI is running
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000
# Expected: HTTP 200

# 9. End-to-end RAG query test
curl -s -X POST http://127.0.0.1:8000/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What was discussed at recent board meetings?"}' | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f\"Answer length: {len(r['answer'])} chars\")
print(f\"Citations: {r['chunks_retrieved']}\")
print(f\"Latency: {r['latency_seconds']}s\")
"
# Expected: non-empty answer with citations, ~10-15s latency
```

## 10. Cron Jobs

### Embedding pipeline (runs every 30 minutes)

Picks up new chunks from PostgreSQL and embeds them into Qdrant:

```bash
(crontab -l 2>/dev/null; echo '*/30 * * * * cd /home/qorvault/ksd-boarddocs-rag/embedding_pipeline && source venv/bin/activate && python -m embedding_pipeline >> /home/qorvault/ksd-boarddocs-rag/logs/embedding_pipeline.log 2>&1') | crontab -
```

### Database backup (runs nightly at 2 AM)

```bash
(crontab -l 2>/dev/null; echo '0 2 * * * /home/qorvault/ksd-boarddocs-rag/scripts/backup_databases.sh >> /home/qorvault/ksd-boarddocs-rag/backups/cron.log 2>&1') | crontab -
```

### Create the logs directory

```bash
mkdir -p ~/ksd-boarddocs-rag/logs
```

## Open WebUI Configuration

After all services are running, configure Open WebUI by following the guide in `OPENWEBUI_SETUP.md`. This covers connecting to llama-server, installing the BoardDocs RAG tool, and creating the BoardDocs Assistant model preset.
