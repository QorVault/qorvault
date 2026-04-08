#!/usr/bin/env python3
"""Extract and rank the most frequently mentioned person names
across Kent School District board meeting documents.

Uses spaCy NER to find PERSON entities in chunk text, then aggregates
by normalized name with document diversity and temporal context.

Usage:
    python scripts/name_frequency.py
    python scripts/name_frequency.py --since 2025-01-01 --top 25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration — credentials loaded from environment variables (.env)
# ---------------------------------------------------------------------------


def _build_db_config() -> dict:
    """Build psycopg2 connection kwargs from POSTGRES_* env vars."""
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. " "Copy .env.example to .env and fill in credentials."
        )
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "qorvault"),
        "password": password,
        "dbname": os.environ.get("POSTGRES_DB", "qorvault"),
    }


DB_CONFIG = _build_db_config()

TENANT_ID = "kent_sd"

# Names that spaCy frequently misclassifies as PERSON
FALSE_POSITIVES = {
    "board",
    "director",
    "superintendent",
    "president",
    "council",
    "committee",
    "state",
    "washington",
    "kent",
    "district",
    "school",
    "staff",
    "public",
    "speaker",
    "member",
}

MIN_NAME_LEN = 4
MAX_NAME_LEN = 40

QUERY = """
SELECT c.id AS chunk_id,
       c.content,
       d.title,
       d.meeting_date,
       d.document_type,
       d.id AS document_id
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.tenant_id = %s
  AND d.meeting_date >= %s
ORDER BY d.meeting_date DESC, c.chunk_index
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_password(config: dict) -> str:
    """Return a connection string with the password masked."""
    return (
        f"host={config['host']} port={config['port']} " f"user={config['user']} password=*** dbname={config['dbname']}"
    )


def load_spacy():
    """Load the best available spaCy model."""
    import spacy  # noqa: delay import so connection errors fail fast

    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
            if model_name != "en_core_web_lg":
                print(
                    f"WARNING: en_core_web_lg not available, " f"falling back to {model_name} (lower accuracy)",
                    file=sys.stderr,
                )
            else:
                print(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            continue

    print(
        "ERROR: No spaCy English model found. Tried: en_core_web_lg, en_core_web_sm",
        file=sys.stderr,
    )
    print("Install with: python -m spacy download en_core_web_lg", file=sys.stderr)
    sys.exit(1)


def normalize_name(name: str) -> str | None:
    """Normalize a name or return None if it should be filtered out."""
    name = name.strip()
    if not name:
        return None
    # Title-case normalization
    name = name.title()
    # Length filters
    if len(name) < MIN_NAME_LEN or len(name) > MAX_NAME_LEN:
        return None
    # False-positive filter (check each word independently)
    if name.lower() in FALSE_POSITIVES:
        return None
    # Single-word names that match false positives
    words = name.split()
    if len(words) == 1 and words[0].lower() in FALSE_POSITIVES:
        return None
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract and rank person names from board meeting documents.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default="2024-09-01",
        help="Start date in ISO format (default: 2024-09-01, school year start)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top names to display (default: 50)",
    )
    args = parser.parse_args()

    try:
        since_date = date.fromisoformat(args.since)
    except ValueError:
        print(f"ERROR: Invalid date format: {args.since} (use YYYY-MM-DD)", file=sys.stderr)
        sys.exit(1)

    # -- Connect to database --
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed", file=sys.stderr)
        sys.exit(1)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
    except Exception as exc:
        print("ERROR: Database connection failed", file=sys.stderr)
        print(f"  Connection: {mask_password(DB_CONFIG)}", file=sys.stderr)
        print(f"  Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']})")

    # -- Fetch chunks --
    print(f"Querying chunks with meeting_date >= {since_date} ...")
    cur.execute(QUERY, (TENANT_ID, since_date))
    rows = cur.fetchall()
    total_chunks = len(rows)
    print(f"Found {total_chunks:,} chunks to process")

    if total_chunks == 0:
        print("No chunks found for the specified date range.")
        cur.close()
        conn.close()
        return

    # -- Load spaCy --
    nlp = load_spacy()

    # -- Process chunks --
    # name -> total count
    name_counts: dict[str, int] = defaultdict(int)
    # name -> set of document_ids
    name_docs: dict[str, set] = defaultdict(set)
    # name -> most recent meeting date
    name_latest: dict[str, date] = {}
    # name -> {doc_type -> count}
    name_by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    errors = []
    t_start = time.time()

    for i, (chunk_id, content, title, meeting_date, doc_type, doc_id) in enumerate(rows):
        # Progress indicator every 500 chunks
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (total_chunks - i) / rate if rate > 0 else 0
            print(
                f"  Processed {i:,}/{total_chunks:,} chunks "
                f"({i*100//total_chunks}%) — "
                f"{rate:.0f} chunks/sec, ETA {eta:.0f}s"
            )

        if not content or not content.strip():
            continue

        try:
            doc = nlp(content)
        except Exception as exc:
            errors.append((str(chunk_id), str(exc)))
            continue

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name = normalize_name(ent.text)
            if name is None:
                continue

            name_counts[name] += 1
            name_docs[name].add(str(doc_id))

            if meeting_date and (name not in name_latest or meeting_date > name_latest[name]):
                name_latest[name] = meeting_date

            dtype = doc_type or "unknown"
            name_by_type[name][dtype] += 1

    elapsed = time.time() - t_start
    cur.close()
    conn.close()

    # -- Sort and display results --
    ranked = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nProcessed {total_chunks:,} chunks in {elapsed:.1f}s " f"({total_chunks/elapsed:.0f} chunks/sec)")
    if errors:
        print(f"Skipped {len(errors)} chunk(s) due to errors:")
        for cid, err in errors[:5]:
            print(f"  chunk {cid}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    print(f"Unique names found: {len(ranked):,}")

    # -- Main table --
    top_n = min(args.top, len(ranked))
    print(f"\n{'='*80}")
    print(f"TOP {top_n} PERSON NAMES — meetings since {since_date}")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Name':<30}{'Mentions':>10}{'Documents':>11}{'Latest Meeting':>16}")
    print(f"{'-'*6}{'-'*30}{'-'*10}{'-'*11}{'-'*16}")

    for rank, (name, count) in enumerate(ranked[:top_n], 1):
        n_docs = len(name_docs[name])
        latest = str(name_latest.get(name, "N/A"))
        print(f"{rank:<6}{name:<30}{count:>10,}{n_docs:>11}{latest:>16}")

    # -- Breakdown by document type for top 10 --
    top_10 = ranked[:10]
    if top_10:
        # Collect all document types that appear in the top 10
        all_types = set()
        for name, _ in top_10:
            all_types.update(name_by_type[name].keys())
        all_types = sorted(all_types)

        print(f"\n{'='*80}")
        print("TOP 10 NAMES — BREAKDOWN BY DOCUMENT TYPE")
        print(f"{'='*80}")

        # Header
        type_col_width = max(12, *(len(t) for t in all_types)) if all_types else 12
        header = f"{'Name':<25}"
        for t in all_types:
            header += f"{t:>{type_col_width}}"
        print(header)
        print("-" * len(header))

        for name, total in top_10:
            row = f"{name:<25}"
            for t in all_types:
                c = name_by_type[name].get(t, 0)
                row += f"{c:>{type_col_width}}"
            row += f"  (total: {total})"
            print(row)

    print()


if __name__ == "__main__":
    main()
