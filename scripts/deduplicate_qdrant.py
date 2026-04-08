#!/usr/bin/env python3
"""Deduplicate the boarddocs_chunks Qdrant collection.

This script performs safe, auditable deduplication in three phases:

  Phase 1 (ANALYSIS) — Read-only scan of all points. Groups by
      (SHA-256 of content, title) to find true duplicates. Also
      identifies low-value chunks and policy revision history.
      Always runs.

  Phase 2 (DEDUPLICATION) — Deletes true duplicates and low-value
      chunks. Only runs when --execute is passed.

  Phase 3 (VERIFICATION) — Re-scans the collection, confirms zero
      remaining duplicates, runs test queries, and verifies that
      known policy revisions were preserved. Runs after Phase 2.

Definitions:
  TRUE DUPLICATE  — Two points with identical SHA-256(content) AND
                    identical title. Same file ingested multiple times.
  REVISION        — Points with similar titles but different content
                    hashes across different meeting dates. These are
                    the legislative revision history and are NEVER deleted.
  LOW-VALUE CHUNK — Content with <50 non-whitespace characters, or
                    content that is only "This Page Intentionally Blank".

Usage:
  # Analysis only (safe, read-only):
  python deduplicate_qdrant.py

  # Execute deduplication:
  python deduplicate_qdrant.py --execute
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, PointIdsList

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "boarddocs_chunks"
BATCH_SIZE = 1000
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds

SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_PATH = SCRIPTS_DIR / "dedup_analysis.json"
DELETIONS_PATH = SCRIPTS_DIR / "dedup_deletions.json"

# Low-value detection thresholds
MIN_SUBSTANTIVE_CHARS = 50
BLANK_PAGE_PATTERNS = [
    re.compile(r"^\s*(this\s+page\s+(is\s+)?intentionally\s+(left\s+)?blank)\s*$", re.IGNORECASE),
    re.compile(r"^\s*intentionally\s+blank\s*$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def content_hash(text: str) -> str:
    """SHA-256 hex digest of the content string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_first_markdown_header(text: str) -> str:
    """Remove only the FIRST markdown header line (the document title).

    Previous version stripped ALL header-formatted lines, which caused
    false positives — chunks where substantive content happened to be
    formatted as ## headers were incorrectly counted as having 0
    non-whitespace chars.  Now we only strip the leading title line.
    """
    lines = text.strip().splitlines()
    if lines and re.match(r"^\s*#{1,6}\s", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def non_whitespace_len(text: str) -> int:
    """Count non-whitespace characters."""
    return len(re.sub(r"\s", "", text))


def is_blank_page_only(text: str) -> bool:
    """Return True if the only substantive content is a blank-page notice."""
    cleaned = strip_first_markdown_header(text).strip()
    return any(pat.match(cleaned) for pat in BLANK_PAGE_PATTERNS)


def is_header_only(text: str, title: str | None) -> bool:
    """Return True if the chunk is just a repeated document title/header."""
    if not title:
        return False
    cleaned = strip_first_markdown_header(text).strip()
    # After stripping headers, if what's left matches the title (case-insensitive),
    # or is empty, this is a header-only chunk.
    if not cleaned:
        return True
    return cleaned.lower().strip() == title.lower().strip()


def scroll_all_points(client: QdrantClient) -> list:
    """Scroll through every point in the collection.

    Returns a list of (point_id, payload) tuples.
    Uses with_vectors=False to minimize memory and bandwidth.
    """
    all_points = []
    offset = None
    batch_num = 0

    # Get total count for progress reporting.
    collection_info = client.get_collection(COLLECTION)
    total_points = collection_info.points_count
    print(f"\nCollection has {total_points:,} points. Scrolling in batches of {BATCH_SIZE}...")

    while True:
        batch_num += 1
        scanned = len(all_points)
        pct = (scanned / total_points * 100) if total_points else 0
        print(f"  Batch {batch_num} — {scanned:,} / {total_points:,} scanned ({pct:.1f}%)", end="\r")

        points, next_offset = _scroll_with_retry(client, offset)

        for pt in points:
            all_points.append((pt.id, pt.payload))

        if next_offset is None:
            break
        offset = next_offset

    print(f"  Batch {batch_num} — {len(all_points):,} / {total_points:,} scanned (100.0%)   ")
    return all_points


def _scroll_with_retry(client: QdrantClient, offset):
    """Scroll one batch with retry logic for network timeouts."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.scroll(
                collection_name=COLLECTION,
                offset=offset,
                limit=BATCH_SIZE,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"\n  ⚠ Scroll failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)


def parse_meeting_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD meeting date, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Phase 1 — Analysis
# ---------------------------------------------------------------------------
def phase1_analysis(client: QdrantClient) -> dict:
    """Read-only analysis of the entire collection.

    Returns a dict with all analysis results (also saved to disk).
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — ANALYSIS (read-only)")
    print("=" * 70)

    all_points = scroll_all_points(client)
    total_scanned = len(all_points)
    print(f"\nTotal points scanned: {total_scanned:,}")

    # ------------------------------------------------------------------
    # 1a. Group by (content_hash, title) for true-duplicate detection
    # ------------------------------------------------------------------
    print("\nGrouping points by (content_hash, title)...")
    groups: dict[tuple[str, str], list] = defaultdict(list)
    hash_by_title: dict[str, set[str]] = defaultdict(set)  # title → set of hashes

    for point_id, payload in all_points:
        content = payload.get("content", "")
        title = payload.get("title") or ""
        chash = content_hash(content)

        groups[(chash, title)].append(
            {
                "point_id": str(point_id),
                "title": title,
                "meeting_date": payload.get("meeting_date"),
                "chunk_index": payload.get("chunk_index"),
                "content_preview": content[:150],
                "content_length": len(content),
            }
        )

        # Track distinct hashes per title for revision detection.
        if title:
            hash_by_title[title].add(chash)

    unique_combos = len(groups)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    num_dup_groups = len(dup_groups)
    total_removable = sum(len(v) - 1 for v in dup_groups.values())

    print(f"  Unique (content_hash, title) combinations: {unique_combos:,}")
    print(f"  Duplicate groups (count > 1):               {num_dup_groups:,}")
    print(f"  Total removable duplicate points:           {total_removable:,}")

    # Top 20 most-duplicated groups.
    top20_dup = sorted(dup_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]
    print("\n  Top 20 most-duplicated groups:")
    for i, ((chash, title), members) in enumerate(top20_dup, 1):
        dates = sorted(set(m["meeting_date"] or "unknown" for m in members))
        print(f'    {i:>2}. [{len(members)} copies] "{title[:70]}"')
        print(f"        Dates: {', '.join(dates[:5])}{'...' if len(dates) > 5 else ''}")

    # ------------------------------------------------------------------
    # 1b. Low-value chunk detection
    # ------------------------------------------------------------------
    print("\n  Identifying low-value chunks...")
    low_value_short: list[dict] = []  # <50 non-whitespace chars
    low_value_blank: list[dict] = []  # "This Page Intentionally Blank"
    low_value_header: list[dict] = []  # Only a repeated title/header

    for point_id, payload in all_points:
        content = payload.get("content", "")
        title = payload.get("title") or ""
        # Conservative approach: measure the FULL content first.  Only
        # strip the first markdown header (document title) before the
        # secondary check — this avoids false positives where real
        # content is formatted as ## headers.
        stripped = strip_first_markdown_header(content)
        nws_raw = non_whitespace_len(content)
        nws_stripped = non_whitespace_len(stripped)

        info = {
            "point_id": str(point_id),
            "title": title,
            "meeting_date": payload.get("meeting_date"),
            "content_preview": content[:150],
            "non_ws_chars_raw": nws_raw,
            "non_ws_chars_stripped": nws_stripped,
        }

        # A chunk is "short" only if BOTH the raw content AND the
        # title-stripped content have <50 non-whitespace chars.
        # This prevents deleting chunks that have real text in
        # header format.
        if nws_raw < MIN_SUBSTANTIVE_CHARS and nws_stripped < MIN_SUBSTANTIVE_CHARS:
            low_value_short.append(info)
        elif is_blank_page_only(content):
            low_value_blank.append(info)
        elif is_header_only(stripped, title):
            low_value_header.append(info)

    print(f"    Short chunks (<{MIN_SUBSTANTIVE_CHARS} non-ws chars): {len(low_value_short):,}")
    print(f"    Blank-page chunks:                        {len(low_value_blank):,}")
    print(f"    Header-only chunks:                       {len(low_value_header):,}")

    def _show_examples(label: str, items: list[dict], n: int = 10):
        if not items:
            return
        print(f"\n    Examples of {label}:")
        for item in items[:n]:
            preview = item["content_preview"].replace("\n", "\\n")[:80]
            raw = item["non_ws_chars_raw"]
            stripped = item["non_ws_chars_stripped"]
            print(f'      - [{raw} raw / {stripped} stripped chars] "{preview}"')

    _show_examples("short chunks", low_value_short)
    _show_examples("blank-page chunks", low_value_blank)
    _show_examples("header-only chunks", low_value_header)

    # ------------------------------------------------------------------
    # 1c. Revision detection (informational only)
    # ------------------------------------------------------------------
    print("\n  Revision detection summary (informational — revisions are NEVER deleted):")
    revised_titles = {title: hashes for title, hashes in hash_by_title.items() if len(hashes) > 1}
    print(f"    Titles with multiple content versions: {len(revised_titles):,}")

    # For each revised title, gather meeting-date ranges and version counts.
    revision_details: list[dict] = []
    for title, hashes in revised_titles.items():
        # Collect all meeting dates for this title across all hash variants.
        dates = set()
        for (chash, gtitle), members in groups.items():
            if gtitle == title:
                for m in members:
                    if m["meeting_date"]:
                        dates.add(m["meeting_date"])
        revision_details.append(
            {
                "title": title,
                "distinct_versions": len(hashes),
                "meeting_dates": sorted(dates),
                "date_range": f"{min(dates) if dates else '?'} — {max(dates) if dates else '?'}",
            }
        )

    # Top 10 most-revised.
    revision_details.sort(key=lambda r: r["distinct_versions"], reverse=True)
    print("\n    Top 10 most-revised documents:")
    for i, rev in enumerate(revision_details[:10], 1):
        print(f"      {i:>2}. [{rev['distinct_versions']} versions] \"{rev['title'][:65]}\"")
        print(f"          Date range: {rev['date_range']}")

    # ------------------------------------------------------------------
    # Build analysis result and save to disk.
    # ------------------------------------------------------------------
    # Serialize duplicate groups for JSON (convert tuple keys to strings).
    dup_groups_serializable = []
    for (chash, title), members in sorted(dup_groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        dup_groups_serializable.append(
            {
                "content_hash": chash,
                "title": title,
                "count": len(members),
                "removable": len(members) - 1,
                "members": members,
            }
        )

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_points_scanned": total_scanned,
        "unique_content_title_combos": unique_combos,
        "duplicate_groups_count": num_dup_groups,
        "total_removable_duplicates": total_removable,
        "duplicate_groups": dup_groups_serializable,
        "low_value": {
            "short_chunks": {
                "count": len(low_value_short),
                "examples": low_value_short[:10],
            },
            "blank_page_chunks": {
                "count": len(low_value_blank),
                "examples": low_value_blank[:10],
            },
            "header_only_chunks": {
                "count": len(low_value_header),
                "examples": low_value_header[:10],
            },
        },
        "revisions": {
            "titles_with_multiple_versions": len(revised_titles),
            "top_revised": revision_details[:10],
        },
    }

    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n  Analysis saved to {ANALYSIS_PATH}")

    # Return internal data structures Phase 2 needs.
    analysis["_dup_groups"] = dup_groups
    analysis["_low_value_short_ids"] = [item["point_id"] for item in low_value_short]
    analysis["_low_value_blank_ids"] = [item["point_id"] for item in low_value_blank]
    analysis["_all_points"] = all_points  # Needed for deletion metadata lookups.
    return analysis


# ---------------------------------------------------------------------------
# Phase 2 — Deduplication
# ---------------------------------------------------------------------------
def phase2_deduplication(client: QdrantClient, analysis: dict) -> list[dict]:
    """Delete true duplicates and low-value chunks.

    For each duplicate group, keeps the point with the EARLIEST meeting_date
    (ties broken by lowest chunk_index). Deletes everything else.

    Returns the deletion log.
    """
    print("\n" + "=" * 70)
    print("PHASE 2 — DEDUPLICATION (--execute)")
    print("=" * 70)

    dup_groups = analysis["_dup_groups"]
    deletion_log: list[dict] = []
    ids_to_delete: list[str] = []

    # Build a quick lookup: point_id → payload excerpt for logging.
    payload_lookup: dict[str, dict] = {}
    for point_id, payload in analysis["_all_points"]:
        payload_lookup[str(point_id)] = {
            "title": payload.get("title", ""),
            "meeting_date": payload.get("meeting_date"),
            "content_preview": payload.get("content", "")[:100],
            "chunk_index": payload.get("chunk_index"),
        }

    # ------------------------------------------------------------------
    # 2a. True duplicates — keep earliest, delete rest.
    # ------------------------------------------------------------------
    print(f"\n  Processing {len(dup_groups):,} duplicate groups...")
    for (chash, title), members in dup_groups.items():
        # Sort by (meeting_date ASC, chunk_index ASC).  None dates go last.
        def sort_key(m):
            d = parse_meeting_date(m["meeting_date"])
            # None dates sort after everything else.
            date_key = d if d else datetime.max
            idx = m.get("chunk_index") if m.get("chunk_index") is not None else 999999
            return (date_key, idx)

        sorted_members = sorted(members, key=sort_key)
        keeper = sorted_members[0]
        to_remove = sorted_members[1:]

        for m in to_remove:
            pid = m["point_id"]
            ids_to_delete.append(pid)
            deletion_log.append(
                {
                    "point_id": pid,
                    "title": m["title"],
                    "meeting_date": m["meeting_date"],
                    "content_preview": m["content_preview"][:100],
                    "reason": "true-duplicate",
                    "kept_point_id": keeper["point_id"],
                    "kept_meeting_date": keeper["meeting_date"],
                }
            )

    dup_delete_count = len(ids_to_delete)
    print(f"    Duplicate points to delete: {dup_delete_count:,}")

    # ------------------------------------------------------------------
    # 2b. Low-value chunks.
    # ------------------------------------------------------------------
    low_short_ids = set(analysis["_low_value_short_ids"])
    low_blank_ids = set(analysis["_low_value_blank_ids"])

    # Don't double-delete points already marked as duplicates.
    already_deleting = set(ids_to_delete)

    for pid in low_short_ids | low_blank_ids:
        if pid in already_deleting:
            continue
        info = payload_lookup.get(pid, {})
        reason = "low-value-short" if pid in low_short_ids else "low-value-blank-page"
        ids_to_delete.append(pid)
        deletion_log.append(
            {
                "point_id": pid,
                "title": info.get("title", ""),
                "meeting_date": info.get("meeting_date"),
                "content_preview": info.get("content_preview", "")[:100],
                "reason": reason,
                "kept_point_id": None,
            }
        )

    low_delete_count = len(ids_to_delete) - dup_delete_count
    print(f"    Low-value points to delete: {low_delete_count:,}")
    print(f"    Total deletions:            {len(ids_to_delete):,}")

    # ------------------------------------------------------------------
    # Execute batch deletions.
    # ------------------------------------------------------------------
    if not ids_to_delete:
        print("\n  Nothing to delete.")
        return deletion_log

    print("\n  Executing batch deletions...")
    delete_batch_size = 500
    total_batches = (len(ids_to_delete) + delete_batch_size - 1) // delete_batch_size

    for i in range(0, len(ids_to_delete), delete_batch_size):
        batch = ids_to_delete[i : i + delete_batch_size]
        batch_num = i // delete_batch_size + 1
        pct = batch_num / total_batches * 100

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=batch),
                )
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    print(f"\n  ✗ Batch {batch_num} failed after {MAX_RETRIES} attempts: {exc}")
                    raise
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"\n  ⚠ Delete batch {batch_num} failed (attempt {attempt}): {exc}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

        print(f"    Batch {batch_num}/{total_batches} deleted ({pct:.1f}%)", end="\r")

    print(f"    Batch {total_batches}/{total_batches} deleted (100.0%)   ")
    print(f"\n  ✓ Deleted {len(ids_to_delete):,} points.")

    # Save deletion log.
    with open(DELETIONS_PATH, "w") as f:
        json.dump(deletion_log, f, indent=2, default=str)
    print(f"  Deletion log saved to {DELETIONS_PATH}")

    return deletion_log


# ---------------------------------------------------------------------------
# Phase 3 — Verification
# ---------------------------------------------------------------------------
def phase3_verification(client: QdrantClient):
    """Post-deduplication verification:
    - New point count and reduction %.
    - Confirm zero remaining duplicate groups.
    - Run test queries and check for duplicate titles in results.
    - Verify known policy revisions were preserved.
    """
    print("\n" + "=" * 70)
    print("PHASE 3 — VERIFICATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 3a. New totals.
    # ------------------------------------------------------------------
    info = client.get_collection(COLLECTION)
    new_count = info.points_count
    print("\n  Points before deduplication: 168,950")
    print(f"  Points after deduplication:  {new_count:,}")
    reduction = (168_950 - new_count) / 168_950 * 100
    print(f"  Reduction:                   {reduction:.2f}%")

    # ------------------------------------------------------------------
    # 3b. Re-run duplicate detection.
    # ------------------------------------------------------------------
    print("\n  Re-scanning for remaining duplicates...")
    all_points = scroll_all_points(client)
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for _, payload in all_points:
        content = payload.get("content", "")
        title = payload.get("title") or ""
        chash = content_hash(content)
        groups[(chash, title)] += 1

    remaining_dups = {k: v for k, v in groups.items() if v > 1}
    if remaining_dups:
        print(f"  ⚠ WARNING: {len(remaining_dups)} duplicate groups still remain!")
        for (chash, title), count in list(remaining_dups.items())[:5]:
            print(f'    - [{count} copies] "{title[:70]}"')
    else:
        print("  ✓ Zero duplicate groups remain.")

    # ------------------------------------------------------------------
    # 3c. Test queries.
    # ------------------------------------------------------------------
    print("\n  Running test queries...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    except Exception as exc:
        print(f"  ⚠ Could not load sentence-transformers model: {exc}")
        print("    Skipping test queries.")
        model = None

    test_queries = [
        "superintendent salary and compensation",
        "annual budget 2023-2024",
        "student discipline policy",
        "graduation requirements",
        "school closures and consolidation",
    ]

    if model:
        for query_text in test_queries:
            print(f'\n  Query: "{query_text}"')
            query_vector = model.encode(query_text).tolist()

            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=5,
                with_payload=True,
            )

            titles_seen = []
            for pt in results.points:
                title = pt.payload.get("title", "")
                date = pt.payload.get("meeting_date", "")
                score = pt.score
                preview = pt.payload.get("content", "")[:100].replace("\n", " ")
                print(f"    {score:.4f} | {date} | {title[:50]}")
                print(f"           {preview}")
                titles_seen.append(title)

            # Check for duplicate titles in top 5.
            if len(titles_seen) != len(set(titles_seen)):
                duped = [t for t in set(titles_seen) if titles_seen.count(t) > 1]
                print(f"    ⚠ DUPLICATE TITLES in top 5: {duped}")
            else:
                print("    ✓ No duplicate titles in top 5.")

    # ------------------------------------------------------------------
    # 3d. Verify known revisions were preserved.
    # ------------------------------------------------------------------
    print("\n  Verifying preservation of known policy revisions...")

    for policy_label, search_term in [("Policy 3241", "3241"), ("Policy 2410", "2410")]:
        print(f"\n    {policy_label}:")
        matches, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="title", match=MatchText(text=search_term))]),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        dates = set()
        for pt in matches:
            md = pt.payload.get("meeting_date")
            if md:
                dates.add(md)

        if dates:
            print(f"      Distinct meeting dates: {len(dates)}")
            print(f"      Date range: {min(dates)} — {max(dates)}")
            print(f"      Dates: {', '.join(sorted(dates)[:10])}{'...' if len(dates) > 10 else ''}")
        else:
            print("      No matching points found (may not exist in dataset).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate the boarddocs_chunks Qdrant collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete duplicates (Phase 2). Without this flag, only analysis runs.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  BoardDocs Chunks — Qdrant Deduplication Tool")
    print("=" * 70)
    if args.execute:
        print("  Mode: EXECUTE (Phase 1 + Phase 2 + Phase 3)")
        print("  ⚠ This WILL delete data from Qdrant.")
    else:
        print("  Mode: ANALYSIS ONLY (Phase 1)")
        print("  No data will be modified. Pass --execute to delete duplicates.")
    print()

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # Verify connectivity.
    try:
        info = client.get_collection(COLLECTION)
        print(f"  Connected to Qdrant at {QDRANT_URL}")
        print(f"  Collection: {COLLECTION}")
        print(f"  Points:     {info.points_count:,}")
    except Exception as exc:
        print(f"  ✗ Cannot connect to Qdrant at {QDRANT_URL}: {exc}")
        sys.exit(1)

    # Phase 1 — always runs.
    analysis = phase1_analysis(client)

    if not args.execute:
        print("\n" + "=" * 70)
        print("  Analysis complete. Review the results above and in:")
        print(f"    {ANALYSIS_PATH}")
        print("  To execute deduplication, re-run with --execute")
        print("=" * 70)
        return

    # Phase 2 — only with --execute.
    phase2_deduplication(client, analysis)

    # Phase 3 — verify after deletion.
    phase3_verification(client)

    print("\n" + "=" * 70)
    print("  Deduplication complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
