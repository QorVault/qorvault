# QorVault

Source-available retrieval-augmented generation for civic public records,
with mandatory citations.

## What it does

QorVault answers questions about a school district's public meeting record
by searching across two decades of board agendas, minutes, attachments,
and meeting recordings.  Every answer includes inline citations linking
back to the source document, agenda item, or transcript timestamp.  The
synthesis prompt forbids the model from inventing dollar amounts,
percentages, or vote counts not present in the sources, and citation URLs
are constructed exclusively from chunk metadata — never generated or
modified by the LLM.

The current corpus is the Kent School District (Washington) BoardDocs
archive plus the audio recordings of public board meetings.  The
architecture is intentionally generic: any school district, city council,
or county commission that publishes its records via BoardDocs (or any
similar civic publishing platform) can be loaded into the same pipeline
with new tenant configuration and a per-source loader.

## What's in the corpus

- **BoardDocs documents** — 19,775 records spanning 2005–2026: 951 meeting
  agendas, 6,448 individual agenda items, and 12,206 attachments (policy
  documents, financial statements, contracts, presentations, minutes).
  Source: the Kent School District public BoardDocs portal.

- **Audio meeting transcripts** — 397 board meeting recordings (~400
  hours) transcribed with WhisperX and diarized with pyannote 4.0.4 to
  produce per-speaker turn-by-turn text with timestamps.  Speakers are
  resolved to real names by an LLM pass over each meeting's opening roll
  call.  Each chunk carries words-per-minute, mean transcription
  confidence, and interruption metrics so chunks can be searched by
  speaking pattern and dynamics, not only by content.

- **Washington State OSPI data** — Report Card datasets from data.wa.gov
  filtered to the district: enrollment, graduation, assessment,
  attendance, discipline, growth, teacher demographics, teacher
  experience, SQSS, and WaKIDS.  Refreshed annually by re-running the
  download scripts.

## Architecture

Six independently-deployable components plus a shared PostgreSQL/Qdrant
data layer:

```
infrastructure/         PostgreSQL 16 + pgvector and Qdrant 1.9.0 in
                        rootless Podman containers, managed by systemd
                        quadlets so they auto-start on boot.

boarddocs_loader/       Scrapes BoardDocs JSON exports into the
                        documents table.

document_processor/     Two-tier text extraction (pypdf for digital
                        text, Docling + RapidOCR ONNX for scans) and
                        token-aware chunking.

ocr_service/            Subprocess-isolated FastAPI wrapper around the
                        OCR pipeline.  Subprocess isolation prevents
                        zombie thread memory leaks from hung OCR runs
                        — the previous in-process design leaked 42 GB
                        before being rewritten.

embedding_pipeline/     mxbai-embed-large-v1 (1024-dim, ONNX Runtime
                        CPU) over the chunks table, upserting points
                        to Qdrant at ~140-160 chunks/min on CPU.

rag_api/                FastAPI service answering POST /api/v1/query.
                        Hybrid retrieval, authority-weighted scoring,
                        cross-encoder reranking, and mandatory inline
                        citations.

transcription/          Audio ingest pipeline.  Uploads .wav meetings
                        to Cloudflare R2, provisions GPU pods on
                        RunPod, runs WhisperX with built-in pyannote
                        diarization, downloads transcripts, and loads
                        them into the chunks table for embedding.
```

### Retrieval pipeline

A query flows through the following stages:

1. **Query embedding** (~70 ms).  The query is embedded with the same
   `mxbai-embed-large-v1` model used for the corpus, on CPU via ONNX
   Runtime.

2. **Hybrid search** in parallel:
   - **Vector search** in Qdrant against ~230,000 indexed chunks, using
     cosine similarity over the 1024-dim embeddings.
   - **Keyword search** in Postgres using `tsvector` + `plainto_tsquery`
     against the chunk content with tenant scoping.

3. **Reciprocal Rank Fusion** combines the two ranked lists into a
   single candidate set.

4. **Cross-encoder reranking** with `bge-reranker-v2-m3` (278M params,
   MIT license, ONNX int8 quantized) jointly attends to (query, chunk)
   pairs.  The reranker model file is SHA-256 verified at load time so
   a tampered model file fails to start the API rather than silently
   serving manipulated relevance scores.

5. **Authority weighting**.  Each chunk's reranker score is multiplied
   by a per-document-type weight: policies × 1.4, agendas × 1.3,
   minutes × 1.2, attachments × 1.0, transcripts × 0.7.  The transcript
   penalty exists because verbose ASR output produces 40-100× more
   chunks per source document than agenda items, so without weighting
   transcripts dominate retrieval results purely on chunk volume
   regardless of authority.

6. **Truncate to top_k** (default 10) and pass to Claude Opus for
   synthesis with mandatory `[Source N]` inline citations.

### Why these choices

- **PostgreSQL + Qdrant rather than a single vector database.**  The
  source documents have rich relational metadata (tenant, meeting date,
  committee, document type, agenda item ID) that Postgres handles
  natively, while Qdrant's filterable vector search handles the
  embedding side.  Hybrid retrieval needs both engines anyway.

- **ONNX Runtime CPU rather than PyTorch GPU.**  Embedding inference on
  CPU runs at ~140-160 chunks/min, fast enough for the corpus sizes
  this project targets, and avoids GPU-driver instability that affects
  PyTorch loading BERT-family models on some integrated GPUs.  The
  reranker is the same: ~3.5s for 20 candidates with 400+ token
  passages on CPU.

- **Subprocess-isolated OCR.**  Docling and Surya/RapidOCR hold large
  amounts of memory in worker threads.  When an OCR run hangs (which
  happens with malformed PDFs), the parent process can't reliably
  reclaim the memory because Python threads can't be killed.  Running
  OCR in a spawned subprocess that the parent can `SIGKILL` on timeout
  lets the API server stay healthy across thousands of difficult
  documents.

- **Anthropic Claude for synthesis, local Qwen3-30B for chat.**  The
  RAG API uses Claude Opus 4.6 for the final answer synthesis because
  hallucination resistance matters more than throughput for civic
  records.  A local llama.cpp + Qwen3-30B-A3B server is also wired up
  for Open WebUI conversational use that doesn't require strict
  citation discipline.

## Status

The pipeline is built and serving queries against the full Kent SD
corpus:

- **Infrastructure** — PostgreSQL 16 + pgvector and Qdrant 1.9.0 running
  under systemd quadlets, auto-starting on boot.
- **BoardDocs ingest** — 19,775 records loaded; 19,676 fully processed,
  24 deferred, 75 failed (0.4% failure rate, all on corrupt or empty
  source files).
- **Embeddings** — 230,587 points indexed in Qdrant across BoardDocs
  chunks and transcript chunks.
- **RAG API** — serving on `127.0.0.1:8000`.  Typical end-to-end query
  latency is around 13 seconds, dominated by the LLM call (~13 s);
  embedding takes ~70 ms and retrieval ~10 ms.
- **Audio transcription** — 397 meetings ingested with speaker
  resolution via Claude Haiku passes over the opening roll call.

## Honest limitations

- **Single tenant deployed.**  The data model and code support
  multi-tenant operation (every table and Qdrant payload is scoped by
  `tenant_id`), but only the `kent_sd` tenant has actually been loaded
  and tested end-to-end.  Replicating for another district requires
  writing a per-source loader (the BoardDocs JSON shape varies between
  districts) and configuring a new tenant row.

- **No built-in authentication.**  The RAG API binds to `127.0.0.1` by
  default and is intended to run behind a reverse proxy with
  authentication, IP allow-listing, or both.  Do not expose
  `/api/v1/query` directly to the open internet.

- **The LLM-to-SQL routing path is opt-in only and weakly defended.**
  `rag_api/rag_api/database_handler.py` accepts an LLM-generated SQL
  query and runs it against the database with a deny-list validator
  (blocks `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT/
  REVOKE/EXECUTE` plus `;` and `--` comments).  It does not have an
  AST-based table allow-list, does not run under a read-only Postgres
  role, and does not enforce a statement timeout.  The default UI does
  not enable this path; it is opt-in only via an explicit
  `enable_routing: true` field on the API request body.  **Do not
  enable this flag in any deployment that lacks both an authentication
  boundary and a least-privilege database role.**  Hardening this
  handler with an AST allow-list, a `SELECT`-only role on
  `(documents, chunks, tenants)`, and a `statement_timeout` is on the
  post-launch backlog.  See [SECURITY.md](SECURITY.md) for more.

- **Diarization requires GPU.**  CPU diarization with pyannote 4.0.4
  benchmarks at 0.29× realtime — 51 minutes to process 15 minutes of
  audio.  The 397-meeting corpus would take roughly 58 days on four
  CPU workers, which is impractical.  Full transcription was done on
  RunPod GPU pods.  Anyone replicating this pipeline for their own
  district needs either GPU hardware or RunPod credit.

- **OCR service is shipped but currently disabled.**  Document
  processing for the existing corpus is complete, so the OCR service
  is stopped.  Re-enable it via
  `systemctl --user enable --now ksd-ocr-service.service` if you
  ingest new documents that need OCR.

- **The embedding cron is commented out.**  Same reason — the corpus is
  embedded and there is nothing to do until new documents arrive.
  Re-enable it in `crontab -e` when you start adding new documents.

- **A small fraction of source documents could not be processed.**  75
  attachments out of 19,775 (0.4%) are corrupt or empty in the upstream
  BoardDocs export and were not retried.  24 are deferred.

## Quick start

Full step-by-step setup is in
[`infrastructure/README.md`](infrastructure/README.md) for the database
layer and [`RESTORATION.md`](RESTORATION.md) for the full system from a
fresh Fedora install.  The short version:

```bash
# Clone
git clone https://github.com/<your-fork>/qorvault.git
cd qorvault

# Database layer (PostgreSQL + pgvector + Qdrant in Podman)
cd infrastructure
cp .env.example .env
# Edit .env to set POSTGRES_PASSWORD
./setup.sh

# RAG API
cd ../rag_api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Set ANTHROPIC_API_KEY in ../.env or in your shell
uvicorn rag_api.main:app --host 127.0.0.1 --port 8000
```

You will need:

- A self-hosted server with sufficient unified RAM for the embedding
  model and (optionally) a local LLM
- Podman (rootless), Python 3.11+
- An Anthropic API key for query synthesis
- *Optional:* a local llama.cpp server for chat-style queries via
  Open WebUI
- *Optional:* RunPod credit and a Cloudflare R2 bucket for the audio
  transcription pipeline

## Repository layout

```
boarddocs_loader/       BoardDocs JSON → documents table
document_processor/     text extraction + chunking
ocr_service/            subprocess-isolated PDF OCR
embedding_pipeline/     mxbai-embed-large-v1 → Qdrant
rag_api/                FastAPI query endpoint + retrieval pipeline
transcription/          audio → diarized transcripts (RunPod)
infrastructure/         Podman containers + systemd quadlets
research/               analysis notebooks and source data references
scripts/                ingestion, backup, and maintenance utilities
systemd/                user-mode service unit files
diagnostics/            corpus health and OCR coverage analysis
docs/                   project notes and operational logs
SECURITY.md             vulnerability reporting policy
CONTRIBUTING.md         contributor licensing terms (DCO + BSL grant)
LICENSE                 BSL 1.1 license text
```

## License

QorVault is licensed under the [Business Source License 1.1](LICENSE).

This is **not** an Open Source license in the OSI sense.  It is a
source-available license with a delayed Open Source conversion: each
tagged version automatically converts to **AGPL-3.0-or-later** four
years after the date that version is first publicly distributed.

**What you may do without a commercial license:**

- Read, study, and modify the source code.
- Run QorVault for non-commercial purposes — personal use, academic
  research, civic transparency projects that do not generate revenue
  or financial benefit for the operator.
- Redistribute modified versions, subject to the same BSL 1.1 terms.

**What requires a commercial license:**

Any "Commercial Use" as defined in the LICENSE file, including:

- Offering QorVault, or a substantially-incorporating product, as a
  paid service to third parties.
- Providing hosted, managed, or embedded access to QorVault as a
  service.
- Providing consulting, implementation, or deployment services in
  which QorVault is a material component and you receive
  compensation.
- Incorporating QorVault into a revenue-generating product, platform,
  or service.

There are no blanket exemptions.  The same rule applies to
individuals, for-profits, nonprofits, government agencies, educational
institutions, contractors, and consultants alike.

Public agencies, educational institutions, and civic organizations:
standard four-year use agreements are available.  Contact
donald@qorvault.com.

All other licensing inquiries: donald@qorvault.com.

## Security

Vulnerability reporting policy and known limitations:
[SECURITY.md](SECURITY.md).  Report security issues privately to
**donald@qorvault.com**, not via public GitHub issues.

## Contributing

Contributor terms and the Developer Certificate of Origin sign-off
requirement: [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

donald@qorvault.com

Kent School District Board Director, Position 3.  Built QorVault to
do the job the public elected him to do.
