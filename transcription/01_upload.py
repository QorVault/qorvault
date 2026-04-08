#!/usr/bin/env python3
"""Upload audio files to Cloudflare R2 and create a job manifest.

Scans the local audio directory for .opus and .m4a files, uploads them
to the R2 bucket under the audio/ prefix, and creates manifest.json
at the bucket root for the RunPod worker to consume.

Idempotent: re-running skips files already in R2 with matching size.
The manifest preserves statuses (complete/processing/failed) from
previous runs so re-uploading does not reset transcription progress.

Required .env variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

Usage:
    python 01_upload.py              # Upload all files + create manifest
    python 01_upload.py --dry-run    # Print what would happen, no uploads
    python 01_upload.py --verbose    # Debug logging to console
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_DIR = Path("/home/qorvault/projects/ksd_forensic/input/board_meeting_audio/Archive")
AUDIO_EXTENSIONS = {".opus", ".m4a"}
R2_AUDIO_PREFIX = "audio/"
MANIFEST_KEY = "manifest.json"
WORKER_LOCAL_PATH = Path("worker/worker.py")  # relative to script dir
WORKER_R2_KEY = "worker.py"
MANIFEST_VERSION = 1

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,  # 16 MB
    multipart_chunksize=16 * 1024 * 1024,  # 16 MB
    max_concurrency=4,
    use_threads=True,
)

SCRIPT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("upload")
console = Console()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class UploadStats:
    total_files: int = 0
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_uploaded: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config() -> dict[str, str]:
    """Load and validate R2 configuration from .env file."""
    load_dotenv(SCRIPT_DIR / ".env")

    required = [
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
    ]
    config = {}
    missing = []
    for key in required:
        val = os.environ.get(key)
        if not val:
            missing.append(key)
        else:
            config[key] = val

    if missing:
        console.print(f"[red]Missing required .env variables: {', '.join(missing)}[/red]")
        console.print(f"Create {SCRIPT_DIR / '.env'} from .env.example")
        sys.exit(1)

    config["R2_ENDPOINT_URL"] = f"https://{config['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return config


def create_s3_client(config: dict[str, str]):
    """Create boto3 S3 client configured for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=config["R2_ENDPOINT_URL"],
        aws_access_key_id=config["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=config["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload audio files to Cloudflare R2 and create job manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading anything",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging to console",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    """Configure dual logging: file (always DEBUG) + console (WARNING or DEBUG)."""
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"upload_{timestamp}.log"

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_audio_files(audio_dir: Path) -> list[Path]:
    """Find all .opus/.m4a files, sorted newest-first by filename."""
    files = [f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
    files.sort(key=lambda f: f.name, reverse=True)
    return files


# ---------------------------------------------------------------------------
# R2 inventory
# ---------------------------------------------------------------------------


def get_remote_inventory(s3, bucket: str) -> dict[str, int]:
    """Return {key: size_bytes} for all objects under audio/ prefix."""
    inventory = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=R2_AUDIO_PREFIX):
        for obj in page.get("Contents", []):
            inventory[obj["Key"]] = obj["Size"]
    return inventory


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def fetch_existing_manifest(s3, bucket: str) -> dict | None:
    """Download existing manifest.json from R2, or return None."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        logger.info("Fetched existing manifest: %d files", len(data.get("files", [])))
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing manifest found — will create new one")
            return None
        raise


def build_manifest(
    local_files: list[Path],
    existing_manifest: dict | None,
) -> dict:
    """Build manifest preserving statuses from any existing manifest."""
    existing_by_key: dict[str, dict] = {}
    if existing_manifest and "files" in existing_manifest:
        for entry in existing_manifest["files"]:
            existing_by_key[entry["key"]] = entry

    files = []
    for path in local_files:
        key = f"{R2_AUDIO_PREFIX}{path.name}"
        size = path.stat().st_size

        if key in existing_by_key:
            existing = existing_by_key[key]
            if existing.get("size_bytes") == size and existing["status"] != "pending":
                files.append(existing)
                continue

        files.append(
            {
                "key": key,
                "size_bytes": size,
                "status": "pending",
                "worker_id": None,
                "started_at": None,
                "completed_at": None,
                "transcript_key": None,
                "error": None,
            }
        )

    return {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "total_files": len(files),
        "files": files,
    }


def upload_manifest(s3, bucket: str, manifest: dict, dry_run: bool) -> bool:
    """Upload manifest.json to bucket root."""
    if dry_run:
        logger.info("DRY RUN: would upload manifest.json (%d files)", manifest["total_files"])
        return True
    try:
        body = json.dumps(manifest, indent=2).encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=MANIFEST_KEY,
            Body=body,
            ContentType="application/json",
        )
        logger.info("Uploaded manifest.json (%d files)", manifest["total_files"])
        return True
    except Exception as e:
        logger.error("Failed to upload manifest.json: %s", e)
        return False


# ---------------------------------------------------------------------------
# Worker script upload
# ---------------------------------------------------------------------------


def upload_worker_script(s3, bucket: str, dry_run: bool) -> bool:
    """Upload worker/worker.py to bucket root as worker.py, if it exists."""
    worker_path = SCRIPT_DIR / WORKER_LOCAL_PATH
    if not worker_path.exists():
        logger.info("No %s found locally — skipping worker upload", WORKER_LOCAL_PATH)
        return True

    if dry_run:
        logger.info("DRY RUN: would upload %s as %s", WORKER_LOCAL_PATH, WORKER_R2_KEY)
        return True

    try:
        s3.upload_file(
            Filename=str(worker_path),
            Bucket=bucket,
            Key=WORKER_R2_KEY,
            ExtraArgs={"ContentType": "text/x-python"},
        )
        logger.info("Uploaded %s as %s", WORKER_LOCAL_PATH, WORKER_R2_KEY)
        return True
    except Exception as e:
        logger.error("Failed to upload worker.py: %s", e)
        return False


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class _ProgressCallback:
    """Callback for boto3 upload_file that drives a rich progress bar."""

    def __init__(self, progress: Progress, task_id):
        self._progress = progress
        self._task_id = task_id

    def __call__(self, bytes_transferred: int):
        self._progress.update(self._task_id, advance=bytes_transferred)


def upload_one_file(
    s3,
    bucket: str,
    local_path: Path,
    remote_key: str,
    progress: Progress,
) -> bool:
    """Upload a single file to R2 with progress tracking. Returns True on success."""
    file_size = local_path.stat().st_size
    content_type = "audio/ogg" if local_path.suffix.lower() == ".opus" else "audio/mp4"

    task_id = progress.add_task(f"[cyan]{local_path.name}", total=file_size)
    callback = _ProgressCallback(progress, task_id)

    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=remote_key,
            Config=TRANSFER_CONFIG,
            Callback=callback,
            ExtraArgs={"ContentType": content_type},
        )
        progress.update(task_id, description=f"[green]{local_path.name}")
        return True
    except Exception as e:
        progress.update(task_id, description=f"[red]{local_path.name} FAILED")
        logger.error("Upload failed %s: %s", local_path.name, e)
        return False


def upload_files(
    s3,
    bucket: str,
    local_files: list[Path],
    remote_inventory: dict[str, int],
    dry_run: bool,
) -> UploadStats:
    """Upload all files, skipping those already in R2 with matching size."""
    stats = UploadStats(total_files=len(local_files))

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=dry_run,
    )

    with progress:
        for local_path in local_files:
            remote_key = f"{R2_AUDIO_PREFIX}{local_path.name}"
            file_size = local_path.stat().st_size

            # Idempotency: skip if remote object exists with same size
            remote_size = remote_inventory.get(remote_key)
            if remote_size is not None and remote_size == file_size:
                logger.debug(
                    "Skipping %s (already uploaded, %d bytes)",
                    local_path.name,
                    file_size,
                )
                stats.skipped += 1
                continue

            if dry_run:
                action = "RE-UPLOAD" if remote_size is not None else "UPLOAD"
                console.print(
                    f"  [yellow]WOULD {action}[/yellow] {local_path.name} " f"({file_size / 1024 / 1024:.1f} MB)"
                )
                stats.uploaded += 1
                stats.bytes_uploaded += file_size
                continue

            logger.info(
                "Uploading %s (%d bytes) -> %s",
                local_path.name,
                file_size,
                remote_key,
            )
            if upload_one_file(s3, bucket, local_path, remote_key, progress):
                stats.uploaded += 1
                stats.bytes_uploaded += file_size
            else:
                stats.failed += 1
                stats.errors.append(local_path.name)

    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(stats: UploadStats, manifest: dict, dry_run: bool) -> None:
    """Print rich summary tables."""
    if dry_run:
        console.print()
        console.print(Panel("[yellow]DRY RUN — no changes made[/yellow]", style="yellow"))

    # Upload stats table
    table = Table(title="Upload Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total audio files", str(stats.total_files))
    table.add_row(
        "Uploaded this run",
        f"[green]{stats.uploaded}[/green]",
    )
    table.add_row(
        "Skipped (already in R2)",
        f"[cyan]{stats.skipped}[/cyan]",
    )
    table.add_row(
        "Failed",
        f"[red]{stats.failed}[/red]" if stats.failed else "0",
    )
    if stats.bytes_uploaded:
        size_gb = stats.bytes_uploaded / 1024 / 1024 / 1024
        if size_gb >= 1.0:
            table.add_row("Data transferred", f"{size_gb:.1f} GB")
        else:
            table.add_row(
                "Data transferred",
                f"{stats.bytes_uploaded / 1024 / 1024:.1f} MB",
            )

    console.print(table)

    # Manifest status table
    status_counts: dict[str, int] = {}
    for f in manifest.get("files", []):
        s = f["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    mt = Table(title="Manifest Status", show_header=True)
    mt.add_column("Status", style="bold")
    mt.add_column("Count", justify="right")

    colors = {
        "pending": "yellow",
        "processing": "blue",
        "complete": "green",
        "failed": "red",
    }
    for status, count in sorted(status_counts.items()):
        color = colors.get(status, "white")
        mt.add_row(status, f"[{color}]{count}[/{color}]")

    console.print(mt)

    if stats.errors:
        console.print(f"\n[red]Failed uploads ({len(stats.errors)}):[/red]")
        for name in stats.errors:
            console.print(f"  [red]x[/red] {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    console.print(Panel("[bold]BoardDocs Audio Upload to Cloudflare R2[/bold]"))

    # Load config and connect
    config = load_config()
    bucket = config["R2_BUCKET_NAME"]
    logger.info("Bucket: %s  Endpoint: %s", bucket, config["R2_ENDPOINT_URL"])

    s3 = create_s3_client(config)

    try:
        s3.head_bucket(Bucket=bucket)
        console.print(f"  [green]Connected to R2 bucket:[/green] {bucket}")
    except ClientError as e:
        console.print(f"[red]Cannot access R2 bucket '{bucket}': {e}[/red]")
        return 1

    # Discover local files
    if not AUDIO_DIR.exists():
        console.print(f"[red]Audio directory not found: {AUDIO_DIR}[/red]")
        return 1

    local_files = discover_audio_files(AUDIO_DIR)
    total_size = sum(f.stat().st_size for f in local_files)
    console.print(f"  [green]Found[/green] {len(local_files)} audio files ({total_size / 1024 / 1024 / 1024:.1f} GB)")

    # Remote inventory
    console.print("  Checking existing uploads in R2...")
    remote_inventory = get_remote_inventory(s3, bucket)
    console.print(f"  [green]Found[/green] {len(remote_inventory)} files already in R2")

    # Existing manifest
    existing_manifest = fetch_existing_manifest(s3, bucket)

    # Upload
    console.print()
    stats = upload_files(s3, bucket, local_files, remote_inventory, args.dry_run)

    # Manifest
    manifest = build_manifest(local_files, existing_manifest)
    if not upload_manifest(s3, bucket, manifest, args.dry_run):
        stats.failed += 1
        stats.errors.append("manifest.json")

    # Worker script
    if not upload_worker_script(s3, bucket, args.dry_run):
        stats.failed += 1
        stats.errors.append("worker.py")

    # Summary
    console.print()
    print_summary(stats, manifest, args.dry_run)

    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
