#!/usr/bin/env python3
"""Download completed WhisperX transcripts from R2 to local filesystem.

Reads manifest.json, finds files with status "complete", and downloads
their transcript JSONs to the local output directory.

Usage:
    python download_transcripts.py
    python download_transcripts.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("/home/qorvault/projects/ksd_forensic/output/whisperx_transcripts")
MANIFEST_KEY = "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download WhisperX transcripts from R2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    load_dotenv(SCRIPT_DIR / ".env")

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]

    # Load manifest
    resp = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)
    manifest = json.loads(resp["Body"].read().decode("utf-8"))

    complete = [f for f in manifest["files"] if f["status"] == "complete"]
    print(f"Manifest: {manifest['total_files']} total, {len(complete)} complete")

    if not complete:
        print("No completed transcripts to download.")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for entry in complete:
        transcript_key = entry.get("transcript_key")
        if not transcript_key:
            continue

        filename = transcript_key.split("/", 1)[-1]
        local_path = args.output_dir / filename

        if local_path.exists():
            skipped += 1
            continue

        try:
            s3.download_file(bucket, transcript_key, str(local_path))
            size_kb = local_path.stat().st_size / 1024
            print(f"  Downloaded: {filename} ({size_kb:.0f}KB)")
            downloaded += 1
        except Exception as e:
            print(f"  FAILED: {filename} — {e}", file=sys.stderr)

    print(f"\nDone: {downloaded} downloaded, {skipped} already existed")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
