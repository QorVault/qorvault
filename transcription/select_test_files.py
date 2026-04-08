#!/usr/bin/env python3
"""Select test files for WhisperX proof-of-concept.

Cross-references audio filenames in the R2 manifest with existing YouTube
transcript JSON files. Selects 3 candidates (short, medium, long) that
have YouTube transcripts available for quality comparison.

Usage:
    python select_test_files.py
    python select_test_files.py --manifest manifest.json   # local manifest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
YOUTUBE_TRANSCRIPT_DIR = Path("/home/qorvault/projects/ksd_forensic/output/transcripts")
MONTH_ABBREV = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_audio_date(filename: str) -> date | None:
    """Extract meeting date from audio filename.

    Audio files end with _M_D_YYYY or _MM_DD_YY before extension.
    E.g. '20240912_-_KSD_Regular_Board_Meeting_-_09_11_24.opus' -> 2024-09-11
    """
    stem = filename.rsplit(".", 1)[0]
    m = re.search(r"(\d{1,2})_(\d{1,2})_(\d{2,4})\s*$", stem)
    if not m:
        return None
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_youtube_date(filename: str) -> date | None:
    """Extract date from YouTube transcript filename.

    E.g. '01FEB2023Meeting.json' -> 2023-02-01
    """
    m = re.match(r"(\d{2})([A-Z]{3})(\d{4})Meeting\.json$", filename)
    if not m:
        return None
    day = int(m.group(1))
    month = MONTH_ABBREV.get(m.group(2))
    year = int(m.group(3))
    if not month:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def load_manifest_from_r2() -> dict:
    """Download manifest.json from R2."""
    import boto3

    load_dotenv(SCRIPT_DIR / ".env")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    resp = s3.get_object(Bucket=os.environ["R2_BUCKET_NAME"], Key="manifest.json")
    return json.loads(resp["Body"].read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select WhisperX test files")
    parser.add_argument("--manifest", type=Path, help="Local manifest.json path")
    args = parser.parse_args()

    # Load manifest
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    else:
        print("Loading manifest from R2...")
        manifest = load_manifest_from_r2()

    # Build date -> audio entry map (pending files only)
    audio_by_date: dict[date, dict] = {}
    for entry in manifest["files"]:
        if entry["status"] != "pending":
            continue
        filename = entry["key"].split("/", 1)[-1]
        d = parse_audio_date(filename)
        if d:
            audio_by_date[d] = entry

    print(f"Pending audio files with parseable dates: {len(audio_by_date)}")

    # Build date set from YouTube transcripts
    youtube_dates: set[date] = set()
    for path in YOUTUBE_TRANSCRIPT_DIR.glob("*Meeting.json"):
        d = parse_youtube_date(path.name)
        if d:
            youtube_dates.add(d)

    print(f"YouTube transcripts with parseable dates: {len(youtube_dates)}")

    # Find overlap
    overlap_dates = sorted(audio_by_date.keys() & youtube_dates)
    print(f"Matching dates (audio + YouTube transcript): {len(overlap_dates)}")

    if not overlap_dates:
        print("No matching files found!")
        return 1

    # Build candidates sorted by file size
    candidates = []
    for d in overlap_dates:
        entry = audio_by_date[d]
        candidates.append((d, entry))
    candidates.sort(key=lambda x: x[1].get("size_bytes", 0))

    # Select short (~35MB / ~30 min), medium (median), long (largest)
    n = len(candidates)
    if n >= 3:
        # Short: closest to 35MB
        short_idx = min(range(n), key=lambda i: abs(candidates[i][1].get("size_bytes", 0) - 35_000_000))
        med_idx = n // 2
        long_idx = n - 1
        # Avoid duplicates
        if med_idx == short_idx:
            med_idx = min(med_idx + 1, n - 1)
        if long_idx == short_idx or long_idx == med_idx:
            long_idx = max(long_idx - 1, 0)
        indices = [short_idx, med_idx, long_idx]
    elif n == 2:
        indices = [0, 1]
    else:
        indices = [0]

    print(f"\nSelected {len(indices)} test files:")
    print(f"{'Label':<8} {'Date':<12} {'Size MB':>8}  {'R2 Key'}")
    print("-" * 80)

    selected_keys = []
    labels = ["Short", "Medium", "Long"]
    for i, idx in enumerate(indices):
        d, entry = candidates[idx]
        size_mb = entry.get("size_bytes", 0) / 1e6
        label = labels[i] if i < len(labels) else f"#{i}"
        print(f"{label:<8} {d.isoformat():<12} {size_mb:>8.1f}  {entry['key']}")
        selected_keys.append(entry["key"])

    # Print command for provisioner
    print("\nTo run test transcription:")
    keys_arg = " ".join(f'"{k}"' for k in selected_keys)
    print("  python 03_provision_simple.py --test")
    print("\nOr manually specify keys:")
    print(f"  --test-keys {keys_arg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
