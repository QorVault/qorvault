
# IMPORTANT! All user prompts requesting approval for action done by Claude Code MUST include language such that a 15 year old with little knowledge of programming principles and best practices can understand what they are approving.
# BoardDocs RAG System — Project Context

## What this project is
A RAG (Retrieval-Augmented Generation) system for Kent School District BoardDocs
public meeting records spanning 2005-2026. Goal is civic transparency tooling
that can be replicated for other school districts and local governments.

## Hardware

This project runs on a self-hosted production server with sufficient unified
RAM for embedding models and local LLM inference.  A separate, lower-spec
processing host is used for ingest jobs that don't need GPU.  Both run Fedora
Linux with rootless Podman.

ONNX Runtime CPU is used for embedding inference rather than PyTorch — some
integrated GPU drivers are unstable for BERT-family models, and CPU inference
is fast enough at the corpus sizes this project targets.

- Models directory: /home/qorvault/models/

## Key paths
- Project root: /home/qorvault/projects/ksd-boarddocs-rag/
- Scraped data: /home/qorvault/projects/ksd_forensic/boarddocs/data/ (1,682 meetings)
- PostgreSQL: 127.0.0.1:5432, database=qorvault, user=qorvault (deployment defaults; override via POSTGRES_DB / POSTGRES_USER)
- Qdrant: 127.0.0.1:6333
- RAG API: 127.0.0.1:8000
- OCR Service (when running): localhost:8001
- ONNX model cache: embedding_pipeline/model_cache/mxbai-embed-large-v1-onnx/

## Git / GitHub
- SSH to GitHub (port 22) does not work on this machine. Always use HTTPS remotes.

## Networking
- ALWAYS use 127.0.0.1, NEVER localhost for Podman services (PostgreSQL, Qdrant)
- Fedora resolves localhost to IPv6 (::1) first; Podman containers only bind IPv4
- OCR service runs natively (not in Podman), so localhost is fine for it

## Infrastructure
- PostgreSQL 16 + pgvector in Podman container (boarddocs-postgres)
- Qdrant v1.9.0 in Podman container (boarddocs-qdrant)
- Both managed by systemd quadlets, auto-start on boot
- Schema: infrastructure/init.sql (apply via `podman exec -i boarddocs-postgres psql -U qorvault -d qorvault`)
- Qdrant collection created via REST API (not in init.sql)

## Tech stack rules
- Python 3.11+, always use python3
- Podman, never Docker
- pip with --break-system-packages for system python, venv for services
- ONNX Runtime for embedding inference (NOT PyTorch — see Hardware section)
- Pydantic v2, not v1
- FastAPI for all HTTP services
- ON CONFLICT DO NOTHING for all database inserts (idempotency)

## Database schema
tenant_id='kent_sd' for all records
Tables: tenants, documents, document_pages, chunks
Vector collection: boarddocs_chunks (1024 dimensions, Cosine, mxbai-embed-large-v1)

## Components built so far
1. infrastructure/ — PostgreSQL + Qdrant (complete, healthy)
2. boarddocs_loader/ — Data loader, 19,775 records ingested (complete)
3. ocr_service/ — Subprocess-isolated extraction (complete, service stopped + disabled)
4. document_processor/ — Text extraction + chunking (complete)
   - 19,676 complete, 24 deferred, 75 failed (0.4%)
   - All document types complete: 6,448 agenda items, 951 agendas, 12,206 attachments
5. embedding_pipeline/ — ONNX Runtime mxbai-embed-large-v1 (complete)
   - 116,667 chunks embedded, 168,950 points in Qdrant
   - Cron disabled (commented out) — re-enable in crontab -e if new docs added
6. rag_api/ — FastAPI + httpx Qdrant REST API + Anthropic Claude, working and tested

## OCR service architecture (stopped, disabled — processing complete)
- systemd unit: ~/.config/systemd/user/ksd-ocr-service.service (disabled, no auto-start)
- Re-enable: `systemctl --user enable --now ksd-ocr-service.service`
- Extraction runs in a **spawned subprocess** (multiprocessing, spawn mode) for isolation
- On timeout: subprocess is killed (SIGKILL) and auto-restarted (~15s startup)
- This prevents zombie threads from leaking memory (previous bug: 42GB from unkillable threads)
- 503 responses during worker restart are expected — OCR client circuit breaker handles them
- GPU hidden entirely (CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="") to prevent PyTorch segfaults
- 2-tier extraction: tier 1 = pypdf digital, tier 2 = Docling + RapidOCR ONNX (CPU)
- Worker stats: /stats endpoint shows worker_restarts count

## Document processing details
- PowerPoint extraction: python-pptx with [Content_Types].xml patching for .ppsx compat
- OCR client has circuit breaker: 3 consecutive failures → 60s cooldown
- OCRUnavailableError leaves docs as 'pending' (not 'failed') for retry when OCR restarts
- asyncpg returns jsonb as strings — always json.loads() before isinstance checks
- Embedding cron: commented out in crontab — re-enable if new documents are added
- Doc processor venv path: document_processor/venv/ (not .venv)

## Failed/deferred documents (99 total, 0.5% of corpus)
- 75 failed attachments — corrupt/empty files, not worth retrying
- 24 deferred attachments

## Resuming after reboot
1. PostgreSQL + Qdrant auto-start via systemd quadlets
2. llama-server auto-starts via systemd
3. RAG API: `cd rag_api && set -a && source ../.env && set +a && source venv/bin/activate && uvicorn rag_api.main:app --host 127.0.0.1 --port 8000`
4. OCR service + embedding cron are disabled (processing complete) — re-enable if needed

## Audio transcription pipeline
- 397 board meeting audio files (2014-2026), ~400 hours total
- Source audio: /home/qorvault/projects/ksd_forensic/input/board_meeting_audio/Archive/ (379 .opus, 17 .m4a)
- Pre-converted WAVs: /home/qorvault/projects/ksd_forensic/input/board_meeting_audio/wav_converted/ (397 files, 105GB, 16kHz mono)
- Audio uploaded to Cloudflare R2 bucket for cloud processing
- Transcription worker: transcription/worker/worker.py (WhisperX on RunPod GPUs)
- WhisperX includes built-in diarization via pyannote — transcription + speaker labeling in one pass

## Diarization — CPU is NOT viable
- diarization/ directory has pyannote.audio 4.0.4 CPU-only setup (Python 3.11 venv)
- **Benchmarked at 0.29x realtime on CPU** — 51 min to process 15 min of audio
- Full corpus would take ~58 days with 4 CPU workers — completely impractical
- pyannote's clustering is O(n^2) in embeddings; long files scale even worse
- WhisperX passes full audio to pyannote unchanged — no speedup over direct pyannote
- pyannote 4.0.4 has NO streaming/online mode; diart (separate pkg) is incompatible with v4
- **Decision: diarization must run on GPU** — either via WhisperX on RunPod or standalone GPU pod
- On GPU (A100/V100): pyannote runs at ~2.5% realtime (~1.5 min per hour of audio)

## Component sequence remaining
7. orchestrator — systemd timer, weekly pipeline automation

## Pipeline run order
1. `boarddocs_loader` — scrape data → documents table
2. `document_processor` — extract text + chunk → chunks table
3. `embedding_pipeline` — embed chunks → Qdrant
4. `rag_api` — query endpoint at /api/v1/query

## RAG API details
- Endpoint: POST /api/v1/query with {"query": "..."}
- Uses httpx REST API for Qdrant (not qdrant-client — v1.9 compat issues)
- Retrieves top 10 chunks by cosine similarity
- LLM: Claude Opus 4.6 via Anthropic SDK
- Returns answer + citations with metadata (title, meeting_date, committee_name, source_url)
- Typical latency: ~13s total (0.07s embed, 0.01s retrieval, ~13s LLM)
- Run: `cd rag_api && source venv/bin/activate && uvicorn rag_api.main:app --host 127.0.0.1 --port 8000`

## MANDATORY: Activity Logging

Every Claude Code session MUST:

1. Be started using the `claude-session` command (alias: `cs`)
   NOT the bare `claude` command. This ensures terminal session
   recording is active.

2. Begin by noting the session start in a brief comment visible
   in the terminal — this gets captured in the session log and
   helps the finalizer identify session boundaries.

3. Before ending any session, run:
   `git add -A && git commit -m "session-end: [brief description of what was accomplished]"`

4. The session finalizer runs automatically on exit when using
   `claude-session`. Do not skip it.

The logging system is read-only from Claude Code's perspective —
it runs externally and requires no action beyond using `claude-session`
instead of `claude`.

### Monitoring infrastructure (runs independently)
- **Git auto-snapshot**: cron job commits changes every 60s
- **Filesystem watcher**: `ksd-fs-watcher.service` logs file events to `logs/filesystem_events.jsonl`
- **Log aggregator**: `ksd-log-aggregator.service` inserts events into `ai_activity_log` table
- **auditd**: kernel-level process auditing with key `ksd_activity`
- **Query tool**: `activity_query` CLI for reviewing logged activity

## Data files and git policy
- Large downloaded datasets (OSPI JSON, PDFs) are excluded from git via .gitignore
- research/ospi_data/ and research/kent_school_data/ contain only scripts and docs in git, not data
- manual_extraction/ (large PDFs) is gitignored, backed up to a separate NAS
- Any new downloaded datasets MUST be added to .gitignore before staging
- See research/DATA_SOURCES.md for download/restore instructions
- All database credentials loaded from .env via python-dotenv, never hardcoded in source

## Important decisions made
- Embedding model: mxbai-embed-large-v1 (1024-dim) via ONNX Runtime CPU
- Embedding speed: ~140-160 chunks/min on ONNX Runtime CPU
- Query LLM: Claude Opus 4.6 via Anthropic API
- Data has two formats: structured (per-item subdirs) and flat (agenda.txt)
- Empty meetings (exec sessions) are valid, not errors
- Single Uvicorn worker for all services (multi-worker breaks state)

## Security rules (enforced by pre-commit hooks and Claude Code hooks)

- NEVER hardcode secrets in source code — use environment variables
- NEVER use eval(), exec(), pickle.load(), yaml.load() without SafeLoader, or subprocess with shell=True
- All SQL MUST use parameterized queries — no f-strings or .format() in SQL
- All HTTP services MUST bind to 127.0.0.1, never 0.0.0.0
- Run `pre-commit run --all-files` before committing

## Documentation standards (enforced by interrogate and ruff)

- ALL public functions, classes, and methods require Google-style docstrings
- Include inline comments explaining WHY for non-obvious decisions
- Target: 80% docstring coverage minimum

## Git workflow

- Create feature branch before work: `git checkout -b claude/type-description`
- Conventional Commits: feat:, fix:, docs:, refactor:, test:, security:
- Atomic commits — one logical change per commit
- NEVER push directly to main or master
