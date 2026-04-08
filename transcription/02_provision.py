#!/usr/bin/env python3
"""Provision a RunPod spot GPU pod and monitor transcription progress.

Creates an 8x NVIDIA B200 spot instance that pulls worker.py from R2
and processes the manifest. Monitors progress by polling manifest.json
from R2 every 60 seconds, displaying a live status panel.

Required .env variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    RUNPOD_API_KEY, HF_TOKEN

Usage:
    python 02_provision.py              # Provision pod + monitor
    python 02_provision.py --dry-run    # Print config without creating pod
    python 02_provision.py --pod-id X   # Skip provisioning, monitor existing pod
    python 02_provision.py --verbose    # Debug logging to console
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import runpod
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from runpod.api.graphql import run_graphql_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_KEY = "manifest.json"

DEFAULT_GPU_TYPE_ID = "NVIDIA B200"
DEFAULT_GPU_COUNT = 8
DEFAULT_BID_PER_GPU = 5.0  # $40 total for 8 GPUs
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_DISK_GB = 150
DEFAULT_CLOUD_TYPE = "SECURE"
POD_NAME = "ksd-transcription"

POLL_INTERVAL_STARTUP = 10  # seconds between status checks while starting
POLL_INTERVAL_MONITOR = 60  # seconds between manifest checks while running
STARTUP_TIMEOUT = 600  # 10 minutes max to wait for RUNNING

logger = logging.getLogger("provision")
console = Console()

# Global for signal handler
_active_pod_id: str | None = None

# ---------------------------------------------------------------------------
# Bootstrap script (runs on the pod)
# ---------------------------------------------------------------------------

WORKER_BOOTSTRAP = """\
#!/bin/bash
set -ex
exec > /var/log/worker_bootstrap.log 2>&1
echo "=== Worker bootstrap started at $(date -u) ==="
pip install -q boto3
mkdir -p /app
cat > /tmp/download_worker.py << 'PYEOF'
import os, boto3
s3 = boto3.client('s3',
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto')
s3.download_file(os.environ['R2_BUCKET_NAME'], 'worker.py', '/app/worker.py')
PYEOF
python3 /tmp/download_worker.py
echo "=== Starting worker at $(date -u) ==="
python3 /app/worker.py
echo "=== Worker finished at $(date -u) ==="
"""


def encode_bootstrap() -> str:
    """Base64-encode the bootstrap script for use in dockerArgs.

    RunPod's PyTorch image uses CMD ["/start.sh"]. Setting dockerArgs
    replaces CMD entirely — our bootstrap runs instead of /start.sh.
    This means no SSH or Jupyter, but the worker runs autonomously.
    """
    encoded = base64.b64encode(WORKER_BOOTSTRAP.encode()).decode()
    return f'bash -c "echo {encoded} | base64 -d | bash"'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config() -> dict[str, str]:
    """Load and validate all required config from .env."""
    load_dotenv(SCRIPT_DIR / ".env")

    required = [
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "RUNPOD_API_KEY",
        "HF_TOKEN",
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

    # Set RunPod API key for the SDK
    runpod.api_key = config["RUNPOD_API_KEY"]

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
        description="Provision RunPod spot GPU pod and monitor transcription",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and mutation without creating pod",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging to console",
    )
    parser.add_argument(
        "--pod-id",
        type=str,
        default=None,
        help="Skip provisioning, monitor an existing pod",
    )
    parser.add_argument(
        "--bid",
        type=float,
        default=DEFAULT_BID_PER_GPU,
        help=f"Bid per GPU in $/hr (default: ${DEFAULT_BID_PER_GPU:.2f})",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=DEFAULT_GPU_COUNT,
        help=f"Number of GPUs (default: {DEFAULT_GPU_COUNT})",
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
    log_file = log_dir / f"provision_{timestamp}.log"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

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
# GPU query
# ---------------------------------------------------------------------------


def query_gpu_info(gpu_type_id: str, gpu_count: int) -> dict:
    """Query RunPod for GPU type details and pricing."""
    try:
        gpu_info = runpod.get_gpu(gpu_type_id, gpu_count)
    except ValueError as e:
        console.print(f"[red]GPU type '{gpu_type_id}' not found: {e}[/red]")
        console.print("Available GPU types:")
        for gpu in runpod.get_gpus():
            console.print(f"  {gpu['id']:20s}  {gpu['displayName']:30s}  {gpu['memoryInGb']}GB")
        sys.exit(1)

    logger.info(
        "GPU: %s (%s), VRAM: %dGB, max: %d GPUs",
        gpu_info["id"],
        gpu_info["displayName"],
        gpu_info["memoryInGb"],
        gpu_info.get("maxGpuCount", 0),
    )
    logger.info(
        "Pricing — secure: $%.2f/hr, spot: $%.2f/hr, min bid: $%.2f",
        gpu_info.get("securePrice") or 0,
        gpu_info.get("secureSpotPrice") or 0,
        (gpu_info.get("lowestPrice") or {}).get("minimumBidPrice") or 0,
    )

    return gpu_info


# ---------------------------------------------------------------------------
# Spot pod creation
# ---------------------------------------------------------------------------


def generate_spot_mutation(
    name: str,
    image_name: str,
    gpu_type_id: str,
    gpu_count: int,
    bid_per_gpu: float,
    container_disk_gb: int,
    cloud_type: str,
    docker_args: str,
    env: dict[str, str],
) -> str:
    """Generate podRentInterruptable GraphQL mutation for spot instance."""
    env_items = [f'{{ key: "{k}", value: "{v}" }}' for k, v in env.items()]
    env_str = ", ".join(env_items)

    # Escape docker_args for GraphQL string
    escaped_args = docker_args.replace("\\", "\\\\").replace('"', '\\"')

    return f"""
    mutation {{
      podRentInterruptable(input: {{
        bidPerGpu: {bid_per_gpu}
        cloudType: {cloud_type}
        gpuCount: {gpu_count}
        gpuTypeId: "{gpu_type_id}"
        containerDiskInGb: {container_disk_gb}
        volumeInGb: 0
        minVcpuCount: 16
        minMemoryInGb: 64
        name: "{name}"
        imageName: "{image_name}"
        dockerArgs: "{escaped_args}"
        startSsh: true
        supportPublicIp: true
        ports: "22/tcp"
        env: [{env_str}]
      }}) {{
        id
        costPerHr
        gpuCount
        memoryInGb
        vcpuCount
        containerDiskInGb
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    """


def create_spot_pod(
    config: dict[str, str],
    gpu_type_id: str,
    gpu_count: int,
    bid_per_gpu: float,
    dry_run: bool,
) -> str | None:
    """Create a spot GPU pod. Returns pod_id or None on dry run."""
    bootstrap_cmd = encode_bootstrap()

    pod_env = {
        "R2_ACCOUNT_ID": config["R2_ACCOUNT_ID"],
        "R2_ACCESS_KEY_ID": config["R2_ACCESS_KEY_ID"],
        "R2_SECRET_ACCESS_KEY": config["R2_SECRET_ACCESS_KEY"],
        "R2_BUCKET_NAME": config["R2_BUCKET_NAME"],
        "HF_TOKEN": config["HF_TOKEN"],
        "RUNPOD_API_KEY": config["RUNPOD_API_KEY"],
    }

    mutation = generate_spot_mutation(
        name=POD_NAME,
        image_name=DEFAULT_IMAGE,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        bid_per_gpu=bid_per_gpu,
        container_disk_gb=DEFAULT_DISK_GB,
        cloud_type=DEFAULT_CLOUD_TYPE,
        docker_args=bootstrap_cmd,
        env=pod_env,
    )

    logger.debug("GraphQL mutation:\n%s", mutation)

    if dry_run:
        console.print("\n[yellow]DRY RUN — GraphQL mutation:[/yellow]")
        # Redact secrets in display
        display_mutation = mutation
        for secret_key in ("R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "HF_TOKEN", "RUNPOD_API_KEY"):
            val = config.get(secret_key, "")
            if val:
                display_mutation = display_mutation.replace(val, "***REDACTED***")
        console.print(display_mutation)
        return None

    console.print("  Creating spot pod...")
    response = run_graphql_query(mutation)

    pod_data = response["data"]["podRentInterruptable"]
    pod_id = pod_data["id"]

    logger.info("Pod created: %s", json.dumps(pod_data, indent=2))
    console.print(f"  [green]Pod created:[/green] {pod_id}")
    console.print(f"  Cost: ${pod_data.get('costPerHr', 0):.2f}/hr")
    console.print(f"  GPUs: {pod_data.get('gpuCount', 0)}")
    console.print(f"  RAM: {pod_data.get('memoryInGb', 0)}GB")
    console.print(f"  Disk: {pod_data.get('containerDiskInGb', 0)}GB")

    return pod_id


# ---------------------------------------------------------------------------
# Pod status polling
# ---------------------------------------------------------------------------


def wait_for_running(pod_id: str) -> dict:
    """Poll pod status until RUNNING or timeout. Returns pod info."""
    console.print(f"\n  Waiting for pod {pod_id} to start...")
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > STARTUP_TIMEOUT:
            console.print(f"[red]Pod did not reach RUNNING within {STARTUP_TIMEOUT}s[/red]")
            console.print("Use --pod-id to re-attach later, or check RunPod dashboard")
            sys.exit(1)

        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            logger.warning("Error querying pod status: %s", e)
            time.sleep(POLL_INTERVAL_STARTUP)
            continue

        status = pod.get("desiredStatus", "UNKNOWN")
        logger.debug("Pod %s status: %s (%.0fs elapsed)", pod_id, status, elapsed)

        if status == "RUNNING":
            console.print(f"  [green]Pod is RUNNING[/green] (took {elapsed:.0f}s)")
            return pod

        if status in ("TERMINATED", "EXITED"):
            console.print(f"[red]Pod terminated before reaching RUNNING: {status}[/red]")
            sys.exit(1)

        # Show waiting indicator
        console.print(
            f"  Status: [yellow]{status}[/yellow] ({elapsed:.0f}s elapsed)",
            end="\r",
        )
        time.sleep(POLL_INTERVAL_STARTUP)


# ---------------------------------------------------------------------------
# Manifest polling
# ---------------------------------------------------------------------------


def fetch_manifest_status(s3, bucket: str) -> dict[str, int]:
    """Fetch manifest from R2 and return status counts."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)
        manifest = json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        logger.warning("Failed to fetch manifest: %s", e)
        return {"error": 1}

    counts: dict[str, int] = {}
    for f in manifest.get("files", []):
        s = f["status"]
        counts[s] = counts.get(s, 0) + 1

    counts["_total"] = len(manifest.get("files", []))
    return counts


# ---------------------------------------------------------------------------
# Monitor display
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def render_status_panel(
    pod_id: str,
    pod_info: dict | None,
    status_counts: dict[str, int],
    start_time: float,
    completion_times: list[float],
) -> Panel:
    """Build a rich Panel with current status."""
    total = status_counts.get("_total", 0)
    complete = status_counts.get("complete", 0)
    processing = status_counts.get("processing", 0)
    pending = status_counts.get("pending", 0)
    failed = status_counts.get("failed", 0)

    elapsed = time.monotonic() - start_time
    cost_per_hr = 0.0
    uptime = 0
    pod_status = "UNKNOWN"

    if pod_info:
        cost_per_hr = pod_info.get("costPerHr", 0) or 0
        uptime = pod_info.get("uptimeSeconds", 0) or 0
        pod_status = pod_info.get("desiredStatus", "UNKNOWN")

    total_cost = cost_per_hr * uptime / 3600

    # ETA calculation
    eta_str = "calculating..."
    if len(completion_times) >= 2 and pending + processing > 0:
        # Average time per file based on completion rate
        time_span = completion_times[-1] - completion_times[0]
        files_in_span = len(completion_times) - 1
        if files_in_span > 0 and time_span > 0:
            avg_per_file = time_span / files_in_span
            remaining = pending + processing
            eta_seconds = avg_per_file * remaining
            eta_str = f"~{format_duration(eta_seconds)}"

    # Build table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Pod ID", pod_id)
    color = {"RUNNING": "green", "STOPPED": "red", "EXITED": "red"}.get(pod_status, "yellow")
    table.add_row("Status", f"[{color}]{pod_status}[/{color}]")

    gpu_name = ""
    if pod_info and pod_info.get("machine"):
        gpu_name = pod_info["machine"].get("gpuDisplayName", "")
    gpu_count = pod_info.get("gpuCount", 0) if pod_info else 0
    if gpu_name:
        table.add_row("GPU", f"{gpu_count}x {gpu_name}")

    table.add_row("Cost/hr", f"${cost_per_hr:.2f}")
    table.add_row("Uptime", format_duration(uptime))
    table.add_row("Total cost", f"[bold]${total_cost:.2f}[/bold]")
    table.add_row("", "")  # spacer

    if total > 0:
        pct = complete / total * 100
        bar_filled = int(pct / 100 * 20)
        bar = "[green]" + "#" * bar_filled + "[/green]" + "." * (20 - bar_filled)
        table.add_row("Complete", f"{complete:>4d} / {total}  {bar}  {pct:.0f}%")
    else:
        table.add_row("Complete", f"{complete}")

    table.add_row("Processing", f"[blue]{processing}[/blue]")
    table.add_row("Pending", f"[yellow]{pending}[/yellow]")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed else "0")
    table.add_row("ETA", eta_str)

    return Panel(table, title="[bold]KSD Transcription Monitor[/bold]", border_style="blue")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def handle_sigint(signum, frame):
    """Handle Ctrl+C — ask whether to terminate the pod."""
    global _active_pod_id
    console.print("\n\n[yellow]Interrupted![/yellow]")

    if _active_pod_id:
        console.print(f"Pod {_active_pod_id} is still running.")
        console.print("[bold]Options:[/bold]")
        console.print("  t = Terminate pod (stop billing)")
        console.print("  l = Leave pod running (re-attach with --pod-id)")
        console.print("  c = Cancel (continue monitoring)")

        try:
            choice = input("\nChoice [t/l/c]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "l"

        if choice == "t":
            console.print(f"Terminating pod {_active_pod_id}...")
            try:
                runpod.terminate_pod(_active_pod_id)
                console.print("[green]Pod terminated.[/green]")
            except Exception as e:
                console.print(f"[red]Failed to terminate: {e}[/red]")
            sys.exit(0)
        elif choice == "l":
            console.print(f"Pod {_active_pod_id} left running.")
            console.print(f"Re-attach with: python 02_provision.py --pod-id {_active_pod_id}")
            sys.exit(0)
        else:
            console.print("Continuing...")
            return
    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------


def monitor_pod(pod_id: str, s3, bucket: str) -> dict[str, int]:
    """Poll manifest + pod status until transcription completes. Returns final counts."""
    global _active_pod_id
    _active_pod_id = pod_id

    start_time = time.monotonic()
    prev_complete = 0
    completion_times: list[float] = []

    console.print(f"\n  Monitoring pod [bold]{pod_id}[/bold] (Ctrl+C for options)\n")

    while True:
        # Fetch current state
        try:
            pod_info = runpod.get_pod(pod_id)
        except Exception as e:
            logger.warning("Error querying pod: %s", e)
            pod_info = None

        status_counts = fetch_manifest_status(s3, bucket)

        # Track completion rate
        current_complete = status_counts.get("complete", 0)
        if current_complete > prev_complete:
            now = time.monotonic()
            # Add one timestamp per newly completed file
            for _ in range(current_complete - prev_complete):
                completion_times.append(now)
            prev_complete = current_complete

        # Render
        panel = render_status_panel(pod_id, pod_info, status_counts, start_time, completion_times)
        console.clear()
        console.print(panel)

        # Check exit conditions
        total = status_counts.get("_total", 0)
        done = status_counts.get("complete", 0) + status_counts.get("failed", 0)

        if total > 0 and done >= total:
            console.print("\n[green bold]All files processed![/green bold]")
            return status_counts

        if pod_info:
            pod_status = pod_info.get("desiredStatus", "")
            if pod_status in ("STOPPED", "EXITED", "TERMINATED"):
                console.print(
                    f"\n[red]Pod status is {pod_status} — "
                    f"transcription may be incomplete ({done}/{total} done)[/red]"
                )
                return status_counts

        logger.debug(
            "Poll: complete=%d processing=%d pending=%d failed=%d",
            status_counts.get("complete", 0),
            status_counts.get("processing", 0),
            status_counts.get("pending", 0),
            status_counts.get("failed", 0),
        )

        time.sleep(POLL_INTERVAL_MONITOR)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    pod_id: str,
    status_counts: dict[str, int],
    pod_info: dict | None,
    elapsed: float,
) -> None:
    """Print final summary table."""
    table = Table(title="Transcription Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    total = status_counts.get("_total", 0)
    table.add_row("Total files", str(total))
    table.add_row("Complete", f"[green]{status_counts.get('complete', 0)}[/green]")
    table.add_row("Failed", f"[red]{status_counts.get('failed', 0)}[/red]")

    if pod_info:
        cost_per_hr = pod_info.get("costPerHr", 0) or 0
        uptime = pod_info.get("uptimeSeconds", 0) or 0
        total_cost = cost_per_hr * uptime / 3600
        table.add_row("Total cost", f"${total_cost:.2f}")
        table.add_row("Pod uptime", format_duration(uptime))
    else:
        table.add_row("Elapsed (local)", format_duration(elapsed))

    table.add_row("Pod ID", pod_id)

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # Install signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    console.print(Panel("[bold]KSD Transcription — RunPod Provisioner[/bold]"))

    # Load config
    config = load_config()
    bucket = config["R2_BUCKET_NAME"]
    logger.info("Bucket: %s  Endpoint: %s", bucket, config["R2_ENDPOINT_URL"])

    # R2 client for manifest polling
    s3 = create_s3_client(config)
    try:
        s3.head_bucket(Bucket=bucket)
        console.print(f"  [green]R2 bucket:[/green] {bucket}")
    except ClientError as e:
        console.print(f"[red]Cannot access R2 bucket '{bucket}': {e}[/red]")
        return 1

    # Check manifest exists (skip for dry-run and re-attach)
    manifest_status = fetch_manifest_status(s3, bucket)
    if "error" in manifest_status:
        if args.dry_run:
            console.print("  [yellow]Manifest not found (OK for dry-run)[/yellow]")
        elif args.pod_id:
            console.print("  [yellow]Manifest not found yet — will poll during monitoring[/yellow]")
        else:
            console.print("[red]No manifest.json found in R2. Run 01_upload.py first.[/red]")
            return 1
    else:
        total = manifest_status.get("_total", 0)
        pending = manifest_status.get("pending", 0)
        console.print(f"  [green]Manifest:[/green] {total} files ({pending} pending)")

        if pending == 0 and not args.pod_id:
            console.print("[yellow]No pending files to transcribe.[/yellow]")
            return 0

    # GPU info
    console.print(f"\n  Querying GPU type: {args.gpu_count}x {DEFAULT_GPU_TYPE_ID}...")
    gpu_info = query_gpu_info(DEFAULT_GPU_TYPE_ID, args.gpu_count)
    console.print(f"  [green]GPU:[/green] {gpu_info['displayName']} " f"({gpu_info['memoryInGb']}GB VRAM)")

    on_demand = gpu_info.get("securePrice") or 0
    spot_price = gpu_info.get("secureSpotPrice") or 0
    min_bid = (gpu_info.get("lowestPrice") or {}).get("minimumBidPrice") or 0
    console.print(
        f"  Pricing — on-demand: ${on_demand:.2f}/hr, " f"spot: ${spot_price:.2f}/hr, min bid: ${min_bid:.2f}/hr"
    )
    console.print(f"  [green]Your bid:[/green] ${args.bid:.2f}/GPU " f"(${args.bid * args.gpu_count:.2f}/hr total)")

    if args.bid < min_bid:
        console.print(f"[red]Bid ${args.bid:.2f} is below minimum ${min_bid:.2f}[/red]")
        return 1

    start_time = time.monotonic()

    if args.pod_id:
        # Re-attach to existing pod
        pod_id = args.pod_id
        console.print(f"\n  Re-attaching to pod: {pod_id}")
        try:
            pod_info = runpod.get_pod(pod_id)
            status = pod_info.get("desiredStatus", "UNKNOWN")
            console.print(f"  [green]Pod status:[/green] {status}")
        except Exception as e:
            console.print(f"[red]Cannot reach pod {pod_id}: {e}[/red]")
            return 1
    else:
        # Create new spot pod
        pod_id = create_spot_pod(config, DEFAULT_GPU_TYPE_ID, args.gpu_count, args.bid, args.dry_run)

        if args.dry_run:
            console.print("\n[yellow]DRY RUN — no pod created[/yellow]")
            return 0

        if not pod_id:
            console.print("[red]Failed to create pod[/red]")
            return 1

        # Wait for pod to reach RUNNING
        pod_info = wait_for_running(pod_id)

    # Monitor loop
    final_counts = monitor_pod(pod_id, s3, bucket)

    # Terminate pod
    console.print(f"\n  Terminating pod {pod_id}...")
    try:
        pod_info = runpod.get_pod(pod_id)
        runpod.terminate_pod(pod_id)
        console.print("  [green]Pod terminated[/green]")
    except Exception as e:
        logger.error("Failed to terminate pod: %s", e)
        console.print(f"[red]Failed to terminate pod: {e}[/red]")
        pod_info = None

    elapsed = time.monotonic() - start_time
    print_summary(pod_id, final_counts, pod_info, elapsed)

    failed = final_counts.get("failed", 0)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
