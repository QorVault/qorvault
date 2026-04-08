#!/usr/bin/env python3
"""Deduplicate and rename .eml files into a clean flat directory.

Recursively scans an input directory for .eml files, deduplicates by
Message-ID, and copies exactly one copy of each unique email to an
output directory with a human-readable filename based on date + subject.

Usage:
    python scripts/dedupe_eml.py --input-dir ~/mail/inbox --output-dir ~/mail/clean
    python scripts/dedupe_eml.py --input-dir ~/mail/inbox --output-dir ~/mail/clean --dry-run
"""

from __future__ import annotations

import argparse
import email
import email.policy
import email.utils
import hashlib
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_date(msg: email.message.Message, eml_path: Path) -> datetime:
    """Extract a datetime from the Date header, falling back to file mtime."""
    date_str = msg.get("Date")
    if date_str:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed
    # Fallback: file modification time
    mtime = eml_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=UTC)


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)  # drop special chars
    text = re.sub(r"[\s_]+", "-", text)  # spaces/underscores → hyphens
    text = re.sub(r"-{2,}", "-", text)  # collapse runs of hyphens
    text = text.strip("-")
    return text[:max_len].rstrip("-")


def short_hash(value: str) -> str:
    """Return a 4-character hex hash of a string."""
    return hashlib.md5(value.encode()).hexdigest()[:4]


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes, used when Message-ID is missing."""
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate .eml files into a clean flat directory.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory to search recursively for .eml files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write deduplicated files (created if needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without copying any files",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all .eml files
    eml_files = sorted(input_dir.rglob("*.eml"))
    if not eml_files:
        print(f"No .eml files found in {input_dir}")
        return

    print(f"Found {len(eml_files):,} .eml files in {input_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be copied\n")

    # --- Scan phase: deduplicate by Message-ID ---
    seen_ids: dict[str, Path] = {}  # dedup key → first file seen
    kept: list[tuple[str, Path]] = []  # (dedup_key, path) of kept files
    duplicates: list[tuple[str, Path]] = []  # (dedup_key, path) of skipped dupes
    warnings: list[str] = []
    errors: list[tuple[Path, str]] = []

    t_start = time.time()

    for eml_path in eml_files:
        try:
            raw = eml_path.read_bytes()
        except Exception as exc:
            errors.append((eml_path, f"read error: {exc}"))
            continue

        try:
            msg = email.message_from_bytes(raw, policy=email.policy.default)
        except Exception as exc:
            errors.append((eml_path, f"parse error: {exc}"))
            continue

        # Deduplication key
        message_id = msg.get("Message-ID", "").strip()
        if message_id:
            dedup_key = message_id
        else:
            dedup_key = f"content-hash:{content_hash(raw)}"
            warnings.append(f"No Message-ID: {eml_path} (using content hash)")

        if dedup_key in seen_ids:
            duplicates.append((dedup_key, eml_path))
            continue

        seen_ids[dedup_key] = eml_path
        kept.append((dedup_key, eml_path))

    scan_time = time.time() - t_start
    print(
        f"Scan complete in {scan_time:.1f}s — "
        f"{len(kept):,} unique, {len(duplicates):,} duplicates, "
        f"{len(errors):,} errors"
    )

    # --- Rename phase: build output filenames ---
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, str] = {}  # filename stem → dedup_key that claimed it
    copy_plan: list[tuple[Path, Path]] = []  # (source, dest)
    no_date_count = 0
    no_subject_count = 0

    for dedup_key, eml_path in kept:
        raw = eml_path.read_bytes()
        msg = email.message_from_bytes(raw, policy=email.policy.default)

        # Date
        try:
            dt = parse_date(msg, eml_path)
        except Exception:
            dt = datetime.fromtimestamp(eml_path.stat().st_mtime, tz=UTC)
            no_date_count += 1
            warnings.append(f"Unparseable Date header: {eml_path} (using file mtime)")

        if not msg.get("Date"):
            no_date_count += 1

        date_prefix = dt.strftime("%Y-%m-%d")

        # Subject
        subject = msg.get("Subject", "")
        if not subject or not subject.strip():
            subject = "no-subject"
            no_subject_count += 1

        slug = slugify(subject)
        if not slug:
            slug = "no-subject"

        stem = f"{date_prefix}_{slug}"

        # Disambiguate collisions (different Message-IDs → same filename)
        if stem in used_names and used_names[stem] != dedup_key:
            stem = f"{stem}_{short_hash(dedup_key)}"

        used_names[stem] = dedup_key
        dest = output_dir / f"{stem}.eml"
        copy_plan.append((eml_path, dest))

    # --- Copy phase ---
    copied = 0
    copy_errors: list[tuple[Path, str]] = []

    if not args.dry_run:
        for src, dst in copy_plan:
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as exc:
                copy_errors.append((src, str(exc)))

    # --- Summary ---
    print(f"\n{'='*70}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Input directory:    {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Total .eml found:   {len(eml_files):,}")
    print(f"Unique emails:      {len(kept):,}")
    print(f"Duplicates skipped: {len(duplicates):,}")
    print(f"Parse errors:       {len(errors):,}")
    if not args.dry_run:
        print(f"Files copied:       {copied:,}")
        if copy_errors:
            print(f"Copy errors:        {len(copy_errors):,}")

    if no_date_count or no_subject_count:
        print("\nMissing headers:")
        if no_date_count:
            print(f"  No Date header:    {no_date_count:,} (used file mtime)")
        if no_subject_count:
            print(f"  No Subject header: {no_subject_count:,} (used 'no-subject')")

    # Duplicate breakdown: how many copies of each message?
    from collections import Counter

    dupe_counts = Counter(dk for dk, _ in duplicates)
    if dupe_counts:
        print("\nDuplicate distribution:")
        # Group by copy count
        copies_dist = Counter(dupe_counts.values())
        for n_copies, n_messages in sorted(copies_dist.items()):
            print(f"  {n_messages:,} message(s) had {n_copies} extra cop{'y' if n_copies == 1 else 'ies'}")

    if errors:
        print("\nParse errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if copy_errors:
        print("\nCopy errors:")
        for path, err in copy_errors[:10]:
            print(f"  {path}: {err}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if args.dry_run:
        print("\nFirst 20 planned renames:")
        for src, dst in copy_plan[:20]:
            print(f"  {src.name}")
            print(f"    → {dst.name}")

    print()


if __name__ == "__main__":
    main()
