# BoardDocs RAG Infrastructure

Data services layer for the Kent School District BoardDocs RAG system. Runs PostgreSQL 16 with pgvector and Qdrant on Podman (rootless).

## Prerequisites

- Fedora Linux with Podman installed
- Python 3.10+ with pip
- podman-compose (`sudo dnf install podman-compose`)

Verify tooling:

```bash
podman --version          # 4.0+ required
podman-compose version    # 1.0+ required
python3 --version         # 3.10+ required
```

## Quick Start

```bash
cd ~/ksd-boarddocs-rag/infrastructure

# Make scripts executable
chmod +x setup.sh teardown.sh quadlets/install_quadlets.sh

# Run the full setup (idempotent — safe to run multiple times)
./setup.sh
```

`setup.sh` performs the following steps:
1. Creates `.env` from `.env.example` if it doesn't exist
2. Installs Python dependencies (`psycopg2-binary`, `requests`)
3. Pulls container images (`pgvector/pgvector:pg16`, `qdrant/qdrant:v1.9.0`)
4. Starts containers via `podman-compose`
5. Applies the database schema (extensions, tables, indexes, seed data)
6. Creates the Qdrant `boarddocs_chunks` collection with payload indexes
7. Installs Podman quadlet unit files for auto-start on login
8. Runs all 10 verification checks

## Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env to change passwords or connection settings
```

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_PASSWORD` | `CHANGE_ME_ON_FIRST_DEPLOY` | PostgreSQL password for the application user |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `qorvault` | Database name (deployment choice — rename if desired) |
| `POSTGRES_USER` | `qorvault` | Database user (deployment choice — rename if desired) |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_HTTP_PORT` | `6333` | Qdrant HTTP API port |
| `QDRANT_GRPC_PORT` | `6334` | Qdrant gRPC port |

## Verification

Run the 10-point verification independently:

```bash
python3 verify_infrastructure.py
```

Checks:
1. PostgreSQL TCP port reachable
2. psycopg2 connects to application database
3. pgvector extension installed
4. All four tables exist (tenants, documents, document_pages, chunks)
5. kent_sd tenant row exists
6. documents.ocr_confidence column is double precision
7. Qdrant HTTP health check
8. Qdrant collection boarddocs_chunks exists with vector size 1024
9. Qdrant payload index on tenant_id
10. Both containers running

## Services

| Service | Host Port | Container |
|---|---|---|
| PostgreSQL | 127.0.0.1:5432 | boarddocs-postgres |
| Qdrant HTTP | 127.0.0.1:6333 | boarddocs-qdrant |
| Qdrant gRPC | 127.0.0.1:6334 | boarddocs-qdrant |

All services bind to `127.0.0.1` (loopback only).  If you need LAN access from
another host, terminate inbound traffic at a reverse proxy or SSH tunnel
rather than re-binding the containers to `0.0.0.0`.

## Auto-Start (Quadlets)

Quadlet unit files are installed to `~/.config/containers/systemd/` so containers start automatically when you log in. To check status:

```bash
systemctl --user status boarddocs-postgres
systemctl --user status boarddocs-qdrant
```

To enable lingering (containers start at boot, even before login):

```bash
sudo loginctl enable-linger $USER
```

## Teardown

Stop and remove containers (data volumes are preserved):

```bash
./teardown.sh
```

Stop, remove containers, and delete all data:

```bash
./teardown.sh --volumes
```

## Connecting from the Processing PC

From any machine on the LAN, connect using the RAG Server's IP:

```bash
# PostgreSQL (substitute the user and database from your .env)
psql -h <RAG_SERVER_IP> -U qorvault -d qorvault

# Qdrant
curl http://<RAG_SERVER_IP>:6333/healthz
```

## File Structure

```
infrastructure/
├── podman-compose.yml            # Container definitions
├── init.sql                      # Idempotent database schema
├── setup.sh                      # End-to-end setup script
├── teardown.sh                   # Stop/remove containers
├── verify_infrastructure.py      # 10-check verification
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── quadlets/
│   ├── boarddocs-postgres.container  # Podman quadlet for PostgreSQL
│   ├── boarddocs-qdrant.container    # Podman quadlet for Qdrant
│   └── install_quadlets.sh           # Installs quadlets + reloads systemd
└── README.md
```
