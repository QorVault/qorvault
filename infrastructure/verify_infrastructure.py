#!/usr/bin/env python3
"""BoardDocs RAG Infrastructure Verification
Runs 10 checks and prints [PASS] or [FAIL: reason] for each.
Exit code 0 if all pass, 1 if any fail.
"""

import os
import socket
import subprocess
import sys


def load_env():
    """Load .env file from the infrastructure directory if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


load_env()

PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_DB = os.environ.get("POSTGRES_DB", "qorvault")
PG_USER = os.environ.get("POSTGRES_USER", "qorvault")
PG_PASS = os.environ.get("POSTGRES_PASSWORD")
if not PG_PASS:
    raise RuntimeError("POSTGRES_PASSWORD environment variable is required. See .env.example.")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.environ.get("QDRANT_HTTP_PORT", "6333"))


def check(name, fn):
    """Run a check function, print result, return True/False."""
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL: {e}] {name}")
        return False


# ── Check 1: PostgreSQL TCP port reachable ────────────────────────────
def check_pg_tcp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((PG_HOST, PG_PORT))
    finally:
        sock.close()


# ── Check 2: psycopg2 can connect ────────────────────────────────────
def check_pg_connect():
    import psycopg2

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    conn.close()


# ── Check 3: pgvector extension installed ─────────────────────────────
def check_pgvector():
    import psycopg2

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension not found")
    finally:
        conn.close()


# ── Check 4: All four tables exist ───────────────────────────────────
def check_tables():
    import psycopg2

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    try:
        cur = conn.cursor()
        required = {"tenants", "documents", "document_pages", "chunks"}
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        existing = {row[0] for row in cur.fetchall()}
        missing = required - existing
        if missing:
            raise RuntimeError(f"missing tables: {', '.join(sorted(missing))}")
    finally:
        conn.close()


# ── Check 5: kent_sd tenant exists ───────────────────────────────────
def check_tenant():
    import psycopg2

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tenants WHERE tenant_id = 'kent_sd'")
        if cur.fetchone() is None:
            raise RuntimeError("kent_sd tenant not found")
    finally:
        conn.close()


# ── Check 6: documents.ocr_confidence is double precision ────────────
def check_ocr_column():
    import psycopg2

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='documents' AND column_name='ocr_confidence'"
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("ocr_confidence column not found")
        if row[0] != "double precision":
            raise RuntimeError(f"expected double precision, got {row[0]}")
    finally:
        conn.close()


# ── Check 7: Qdrant health endpoint ─────────────────────────────────
def check_qdrant_health():
    import requests

    resp = requests.get(f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}/healthz", timeout=5)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")


# ── Check 8: Qdrant collection exists with vector size 1024 ─────────
def check_qdrant_collection():
    import requests

    resp = requests.get(
        f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}/collections/boarddocs_chunks",
        timeout=5,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"collection not found (HTTP {resp.status_code})")
    data = resp.json()
    config = data.get("result", {}).get("config", {})
    params = config.get("params", {})
    vectors = params.get("vectors", {})
    # Handle both named and unnamed vector configs
    if isinstance(vectors, dict) and "size" in vectors:
        size = vectors["size"]
    else:
        raise RuntimeError(f"unexpected vectors config: {vectors}")
    if size != 1024:
        raise RuntimeError(f"vector size is {size}, expected 1024")


# ── Check 9: Qdrant payload index on tenant_id ──────────────────────
def check_qdrant_index():
    import requests

    resp = requests.get(
        f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}/collections/boarddocs_chunks",
        timeout=5,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    data = resp.json()
    payload_schema = data.get("result", {}).get("payload_schema", {})
    if "tenant_id" not in payload_schema:
        raise RuntimeError("tenant_id payload index not found")


# ── Check 10: Both containers are running ────────────────────────────
def check_containers():
    result = subprocess.run(
        ["podman", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    running = set(result.stdout.strip().split("\n"))
    required = {"boarddocs-postgres", "boarddocs-qdrant"}
    missing = required - running
    if missing:
        raise RuntimeError(f"not running: {', '.join(sorted(missing))}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    checks = [
        ("PostgreSQL TCP port reachable", check_pg_tcp),
        ("psycopg2 connects to application database", check_pg_connect),
        ("pgvector extension installed", check_pgvector),
        ("All four tables exist", check_tables),
        ("kent_sd tenant row exists", check_tenant),
        ("documents.ocr_confidence is double precision", check_ocr_column),
        ("Qdrant HTTP health check", check_qdrant_health),
        ("Qdrant collection boarddocs_chunks (size 1024)", check_qdrant_collection),
        ("Qdrant payload index on tenant_id", check_qdrant_index),
        ("Both containers running", check_containers),
    ]

    results = [check(name, fn) for name, fn in checks]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
