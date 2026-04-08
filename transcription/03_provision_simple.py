#!/usr/bin/env python3
"""Provision a single-GPU RunPod pod for WhisperX transcription.

Deploys worker_simple.py via R2 bootstrap:
1. Uploads worker_simple.py to R2
2. Creates a spot pod with a bootstrap script that:
   a. Runs /start.sh in the background (preserves web terminal for debugging)
   b. Installs whisperx + boto3
   c. Downloads worker_simple.py from R2
   d. Runs the worker (with --test-keys if in test mode)
   e. Uploads logs to R2 if anything fails
3. Monitors progress by polling manifest.json from R2

Required .env variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    RUNPOD_API_KEY, HF_TOKEN

Usage:
    python 03_provision_simple.py --test         # 3 test files
    python 03_provision_simple.py                # Full run (all pending)
    python 03_provision_simple.py --pod-id X     # Monitor existing pod
    python 03_provision_simple.py --dry-run      # Print config only
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
WORKER_SCRIPT = SCRIPT_DIR / "worker" / "worker_simple.py"
WORKER_R2_KEY = "worker_simple.py"
BOOTSTRAP_LOG_KEY = "logs/bootstrap.log"

# GPU preferences: prioritize High availability + good value
GPU_PREFERENCES = [
    ("NVIDIA A100-SXM4-80GB", 1.22),  # 80GB, High avail, best value
    ("NVIDIA H100 80GB HBM3", 2.69),  # 80GB SXM, High avail
    ("NVIDIA A100 80GB PCIe", 1.14),  # 80GB PCIe, Low avail
    ("NVIDIA H100 PCIe", 2.03),  # 80GB PCIe, Low avail
    ("NVIDIA L40S", 0.71),  # 48GB, Medium avail
    ("NVIDIA GeForce RTX 4090", 0.50),  # 24GB, High avail, cheapest
    ("NVIDIA B200", 4.24),  # 180GB Blackwell, Low avail (needs PyTorch 2.8)
]

# PyTorch 2.4 image: proven on Ampere/Hopper (A100, H100, RTX 4090, L40S).
# PyTorch 2.8 image needed ONLY for Blackwell (B200) — has extra patches.
IMAGE_PYTORCH_24 = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
IMAGE_PYTORCH_28 = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
BLACKWELL_GPUS = {"NVIDIA B200"}
DEFAULT_IMAGE = IMAGE_PYTORCH_24  # overridden to 2.8 if Blackwell GPU selected
DEFAULT_DISK_GB = 80
DEFAULT_CLOUD_TYPE = "ALL"
POD_NAME = "ksd-transcription-simple"

POLL_INTERVAL_STARTUP = 10
POLL_INTERVAL_MONITOR = 30
STARTUP_TIMEOUT = 600

logger = logging.getLogger("provision_simple")
console = Console()

_active_pod_id: str | None = None


# ---------------------------------------------------------------------------
# Bootstrap script template
# ---------------------------------------------------------------------------


def build_bootstrap(test_keys: list[str] | None = None, batch_size: int = 6, no_diarize: bool = False) -> str:
    """Build the bash bootstrap script that runs on the pod.

    Key difference from the old 02_provision.py: we run /start.sh first
    in the background so the web terminal remains available for debugging.
    """
    worker_args = ""
    if test_keys:
        keys_str = " ".join(f'"{k}"' for k in test_keys)
        worker_args += f" --test-keys {keys_str}"
    if batch_size != 6:
        worker_args += f" --batch-size {batch_size}"
    if no_diarize:
        worker_args += " --no-diarize"

    return f"""\
#!/bin/bash
# === KSD Transcription Bootstrap ===
# This script runs as the container CMD. It starts /start.sh in the
# background to preserve web terminal access, then runs the worker.

LOG=/var/log/worker_bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Bootstrap started at $(date -u) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"

# Start the default RunPod entrypoint in background (SSH, Jupyter, etc.)
if [ -f /start.sh ]; then
    echo "Starting /start.sh in background..."
    /start.sh &
    sleep 5
fi

# Upload log helper (called on failure)
upload_log() {{
    python3 -c "
import os, boto3
s3 = boto3.client('s3',
    endpoint_url=f\\"https://{{os.environ['R2_ACCOUNT_ID']}}.r2.cloudflarestorage.com\\",
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto')
with open('$LOG', 'rb') as f:
    s3.put_object(Bucket=os.environ['R2_BUCKET_NAME'],
                  Key='{BOOTSTRAP_LOG_KEY}',
                  Body=f.read(),
                  ContentType='text/plain')
print('Bootstrap log uploaded to R2')
" 2>/dev/null || echo "Failed to upload log to R2"
}}
trap upload_log EXIT

# Install dependencies — protect the image's pre-installed CUDA torch ecosystem
# by using a pip constraints file that pins torch/torchvision/torchaudio
echo "=== Installing dependencies ==="
echo "Pre-installed torch ecosystem:"
python3 -c "import torch; print(f'  torch={{torch.__version__}}')"
python3 -c "import torchvision; print(f'  torchvision={{torchvision.__version__}}')"
python3 -c "import torchaudio; print(f'  torchaudio={{torchaudio.__version__}}')"

# Build constraints file from ALL torch-ecosystem pre-installed packages.
# This prevents pip from upgrading torch, torchvision, torchaudio,
# torchmetrics, lightning, and nvidia CUDA packages. A newer torchmetrics
# triggers a circular import bug in the pre-installed torchvision 0.19.x.
python3 << 'PYEOF' > /tmp/torch_constraints.txt
import subprocess
result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
pin_pkgs = {
    'torch', 'torchvision', 'torchaudio', 'torchmetrics',
    'lightning', 'pytorch-lightning', 'lightning-utilities',
    'torchdata', 'torchtext', 'triton',
}
# Do NOT pin nvidia-cudnn — CTranslate2 needs cuDNN 8 while torch has cuDNN 9.
# Letting pip resolve cudnn versions allows both to coexist.
skip_nvidia = {'nvidia-cudnn-cu12', 'nvidia-cudnn-cu11'}
found = set()
for line in result.stdout.splitlines():
    pkg = line.split('==')[0].lower()
    if pkg in skip_nvidia:
        continue
    if pkg in pin_pkgs or pkg.startswith('nvidia-'):
        print(line)
        found.add(pkg)
# If torchmetrics isn't pre-installed, cap to avoid arniqa circular import
if 'torchmetrics' not in found:
    print('torchmetrics<1.5.0')
PYEOF
echo "Constraints file:"
cat /tmp/torch_constraints.txt

# Install whisperx WITH deps — the constraints file protects torch ecosystem.
# Let whisperx's own dependency resolver pick compatible pyannote.audio version.
pip install -q -c /tmp/torch_constraints.txt \
    boto3 "transformers<4.46" matplotlib whisperx \
    2>&1 | tail -15

# CTranslate2 (used by whisperx/faster-whisper) needs cuDNN 8 runtime
# (libcudnn_ops_infer.so.8). Newer images may bundle cuDNN 9 only.
# IMPORTANT: Do NOT put cuDNN 8 in LD_LIBRARY_PATH — on Blackwell GPUs
# (B200), PyTorch would pick up cuDNN 8 (no Blackwell kernels) instead
# of cuDNN 9. Instead, extract cuDNN 8 to a separate dir and the worker
# will pre-load it via ctypes.CDLL before importing CTranslate2.
echo "=== Checking cuDNN for CTranslate2 ==="
NEED_CUDNN8=0
python3 -c "
import ctypes
try:
    ctypes.CDLL('libcudnn_ops_infer.so.8')
    print('cuDNN 8 already available — skipping download')
except OSError:
    print('cuDNN 8 not found — will download')
    raise SystemExit(1)
" || NEED_CUDNN8=1

if [ "$NEED_CUDNN8" = "1" ]; then
    echo "Installing cuDNN 8 for CTranslate2..."
    pip download "nvidia-cudnn-cu12==8.9.7.29" -d /tmp/cudnn8_dl --no-deps -q 2>&1 | tail -3
    python3 << 'PYEOF'
import zipfile, glob, os
os.makedirs("/opt/cudnn8/lib", exist_ok=True)
whls = glob.glob("/tmp/cudnn8_dl/*.whl")
if not whls:
    print("WARNING: No cuDNN 8 wheel downloaded — CTranslate2 may fail")
else:
    with zipfile.ZipFile(whls[0]) as z:
        for name in z.namelist():
            if "/lib/" in name and (".so.8" in os.path.basename(name)):
                data = z.read(name)
                outpath = f"/opt/cudnn8/lib/{{os.path.basename(name)}}"
                with open(outpath, "wb") as f:
                    f.write(data)
                os.chmod(outpath, 0o755)
    libs = os.listdir("/opt/cudnn8/lib")
    print(f"Extracted {{len(libs)}} cuDNN 8 libraries to /opt/cudnn8/lib")
PYEOF
fi

# Set LD_LIBRARY_PATH: nvidia pip libs + system CUDA only (NO cuDNN 8!)
# cuDNN 8 is loaded explicitly by the worker via ctypes before CTranslate2
export LD_LIBRARY_PATH=$(python3 -c "
import os, glob
paths = set()
for d in glob.glob('/usr/local/lib/python3.*/dist-packages/nvidia/*/lib'):
    paths.add(d)
for d in ['/usr/local/cuda/lib64', '/usr/lib/x86_64-linux-gnu']:
    if os.path.isdir(d):
        paths.add(d)
print(':'.join(sorted(paths)))
"):${{LD_LIBRARY_PATH:-}}
echo "LD_LIBRARY_PATH set (cuDNN 9 + system — cuDNN 8 loaded by worker)"

# Verify torch ecosystem survived
echo "Post-install verification:"
python3 -c "
import torch, torchvision, torchaudio
print(f'  torch={{torch.__version__}} cuda={{torch.cuda.is_available()}}')
print(f'  torchvision={{torchvision.__version__}}')
print(f'  torchaudio={{torchaudio.__version__}}')
import whisperx
print('  whisperx: OK')
"

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
fi

echo "=== Downloading worker from R2 ==="
mkdir -p /app
python3 -c "
import os, boto3
s3 = boto3.client('s3',
    endpoint_url=f\\"https://{{os.environ['R2_ACCOUNT_ID']}}.r2.cloudflarestorage.com\\",
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto')
s3.download_file(os.environ['R2_BUCKET_NAME'], '{WORKER_R2_KEY}', '/app/worker_simple.py')
print('Worker downloaded successfully')
"

if [ $? -ne 0 ]; then
    echo "FATAL: Failed to download worker from R2"
    sleep 3600  # Keep pod alive for debugging via web terminal
    exit 1
fi

echo "=== Starting worker at $(date -u) ==="
python3 -u /app/worker_simple.py{worker_args}
EXIT_CODE=$?

echo "=== Worker finished at $(date -u) with exit code $EXIT_CODE ==="

# Upload final log
upload_log

# Keep pod alive briefly for log collection, then self-terminate
sleep 30
echo "Self-terminating pod..."
python3 -c "
import os, runpod
runpod.api_key = os.environ.get('RUNPOD_API_KEY', '')
pod_id = os.environ.get('RUNPOD_POD_ID', '')
if runpod.api_key and pod_id:
    runpod.terminate_pod(pod_id)
    print(f'Pod {{pod_id}} terminated')
else:
    print('No RUNPOD_API_KEY/POD_ID — skipping self-termination')
" 2>/dev/null || echo "Self-termination failed"
"""


def encode_bootstrap(script: str) -> str:
    """Base64-encode bootstrap for use in dockerArgs."""
    encoded = base64.b64encode(script.encode()).decode()
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
        console.print(f"[red]Missing .env variables: {', '.join(missing)}[/red]")
        console.print(f"Create {SCRIPT_DIR / '.env'} from .env.example")
        sys.exit(1)

    config["R2_ENDPOINT_URL"] = f"https://{config['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    runpod.api_key = config["RUNPOD_API_KEY"]
    return config


def create_s3_client(config: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=config["R2_ENDPOINT_URL"],
        aws_access_key_id=config["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=config["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"provision_simple_{timestamp}.log"

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
# GPU selection
# ---------------------------------------------------------------------------


def select_gpu(override: str | None) -> tuple[str, float]:
    """Select GPU type. Returns (gpu_type_id, bid_per_gpu)."""
    if override:
        for gpu_id, bid in GPU_PREFERENCES:
            if gpu_id == override:
                return gpu_id, bid
        return override, 1.00

    for gpu_id, bid in GPU_PREFERENCES:
        try:
            gpu_info = runpod.get_gpu(gpu_id)
            if gpu_info:
                console.print(f"  [green]GPU available:[/green] {gpu_id}")
                return gpu_id, bid
        except Exception:
            logger.debug("GPU %s not available", gpu_id)
            continue

    console.print("[red]No preferred GPU types available[/red]")
    console.print("Available GPUs:")
    for gpu in runpod.get_gpus():
        console.print(f"  {gpu['id']:30s}  {gpu['memoryInGb']}GB")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Worker upload to R2
# ---------------------------------------------------------------------------


def upload_worker_to_r2(s3, bucket: str) -> None:
    """Upload worker_simple.py to R2 so the pod can download it."""
    if not WORKER_SCRIPT.exists():
        console.print(f"[red]Worker script not found: {WORKER_SCRIPT}[/red]")
        sys.exit(1)

    s3.upload_file(str(WORKER_SCRIPT), bucket, WORKER_R2_KEY)
    size_kb = WORKER_SCRIPT.stat().st_size / 1024
    console.print(f"  Uploaded worker to R2: {WORKER_R2_KEY} ({size_kb:.0f}KB)")


# ---------------------------------------------------------------------------
# Pod creation
# ---------------------------------------------------------------------------


def create_pod(
    config: dict[str, str],
    gpu_type_id: str,
    bid_per_gpu: float,
    docker_args: str,
    dry_run: bool,
    image: str | None = None,
) -> str | None:
    """Create a spot GPU pod with bootstrap. Returns pod_id."""
    image = image or DEFAULT_IMAGE
    pod_env = {
        "R2_ACCOUNT_ID": config["R2_ACCOUNT_ID"],
        "R2_ACCESS_KEY_ID": config["R2_ACCESS_KEY_ID"],
        "R2_SECRET_ACCESS_KEY": config["R2_SECRET_ACCESS_KEY"],
        "R2_BUCKET_NAME": config["R2_BUCKET_NAME"],
        "HF_TOKEN": config["HF_TOKEN"],
        "RUNPOD_API_KEY": config["RUNPOD_API_KEY"],
    }

    env_items = [f'{{ key: "{k}", value: "{v}" }}' for k, v in pod_env.items()]
    env_str = ", ".join(env_items)

    # Escape docker_args for GraphQL string
    escaped_args = docker_args.replace("\\", "\\\\").replace('"', '\\"')

    mutation = f"""
    mutation {{
      podRentInterruptable(input: {{
        bidPerGpu: {bid_per_gpu}
        cloudType: {DEFAULT_CLOUD_TYPE}
        gpuCount: 1
        gpuTypeId: "{gpu_type_id}"
        containerDiskInGb: {DEFAULT_DISK_GB}
        volumeInGb: 0
        minVcpuCount: 4
        minMemoryInGb: 16
        name: "{POD_NAME}"
        imageName: "{image}"
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

    logger.debug("GraphQL mutation:\n%s", mutation)

    if dry_run:
        console.print("\n[yellow]DRY RUN — would create pod with:[/yellow]")
        console.print(f"  GPU: 1x {gpu_type_id}")
        console.print(f"  Bid: ${bid_per_gpu:.2f}/hr")
        console.print(f"  Image: {image}")
        console.print(f"  Disk: {DEFAULT_DISK_GB}GB")
        return None

    console.print("  Creating spot pod...")
    response = run_graphql_query(mutation)

    if "errors" in response:
        console.print(f"[red]RunPod API error: {response['errors']}[/red]")
        return None

    pod_data = response["data"]["podRentInterruptable"]
    pod_id = pod_data["id"]

    logger.info("Pod created: %s", json.dumps(pod_data, indent=2))
    console.print(f"  [green]Pod created:[/green] {pod_id}")
    console.print(f"  Cost: ${pod_data.get('costPerHr', 0):.2f}/hr")

    return pod_id


# ---------------------------------------------------------------------------
# Pod status
# ---------------------------------------------------------------------------


def wait_for_running(pod_id: str) -> dict:
    """Poll until pod is RUNNING or timeout."""
    console.print(f"\n  Waiting for pod {pod_id} to start...")
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > STARTUP_TIMEOUT:
            console.print(f"[red]Pod did not start within {STARTUP_TIMEOUT}s[/red]")
            sys.exit(1)

        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            logger.warning("Error querying pod: %s", e)
            time.sleep(POLL_INTERVAL_STARTUP)
            continue

        status = pod.get("desiredStatus", "UNKNOWN")
        if status == "RUNNING":
            console.print(f"  [green]Pod RUNNING[/green] ({elapsed:.0f}s)")
            return pod
        if status in ("TERMINATED", "EXITED"):
            console.print(f"[red]Pod terminated: {status}[/red]")
            sys.exit(1)

        console.print(f"  Status: [yellow]{status}[/yellow] ({elapsed:.0f}s)", end="\r")
        time.sleep(POLL_INTERVAL_STARTUP)


# ---------------------------------------------------------------------------
# Manifest-based monitoring
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


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def fetch_bootstrap_log(s3, bucket: str) -> str | None:
    """Try to fetch the bootstrap log from R2."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=BOOTSTRAP_LOG_KEY)
        return resp["Body"].read().decode("utf-8", errors="replace")
    except ClientError:
        return None


def monitor_manifest(s3, bucket: str, pod_id: str) -> dict[str, int]:
    """Poll manifest until all files are done or pod stops."""
    global _active_pod_id
    _active_pod_id = pod_id

    console.print(f"\n  Monitoring pod [bold]{pod_id}[/bold] via manifest (Ctrl+C for options)")
    console.print(f"  Polling every {POLL_INTERVAL_MONITOR}s\n")
    start_time = time.monotonic()
    prev_complete = 0
    prev_processing = 0
    no_progress_count = 0

    while True:
        status_counts = fetch_manifest_status(s3, bucket)
        total = status_counts.get("_total", 0)
        complete = status_counts.get("complete", 0)
        processing = status_counts.get("processing", 0)
        pending = status_counts.get("pending", 0)
        failed = status_counts.get("failed", 0)
        done = complete + failed

        elapsed = time.monotonic() - start_time

        if complete > prev_complete:
            new = complete - prev_complete
            no_progress_count = 0
            console.print(
                f"  [{format_duration(elapsed)}] "
                f"[green]+{new} complete[/green] — "
                f"{complete}/{total} done, {pending} pending, {failed} failed"
            )
            prev_complete = complete
        elif processing != prev_processing:
            no_progress_count = 0
            console.print(
                f"  [{format_duration(elapsed)}] "
                f"{processing} processing, {complete}/{total} done, {pending} pending"
            )
            prev_processing = processing
        else:
            no_progress_count += 1
            console.print(
                f"  [{format_duration(elapsed)}] " f"waiting... ({complete}/{total} done, {processing} processing)",
                end="\r",
            )

        # After 10 minutes with no manifest change, check bootstrap log
        if no_progress_count > 0 and no_progress_count % 20 == 0:
            log_content = fetch_bootstrap_log(s3, bucket)
            if log_content:
                console.print("\n  [yellow]Bootstrap log (last 10 lines):[/yellow]")
                for line in log_content.strip().splitlines()[-10:]:
                    console.print(f"    {line}")

        if total > 0 and done >= total:
            console.print(f"\n\n[green bold]All {total} files processed![/green bold]")
            console.print(f"  Complete: {complete}, Failed: {failed}")
            return status_counts

        # Check if pod is still alive
        try:
            pod_info = runpod.get_pod(pod_id)
            pod_status = pod_info.get("desiredStatus", "")
            if pod_status in ("STOPPED", "EXITED", "TERMINATED"):
                console.print(f"\n[red]Pod {pod_status} — {done}/{total} done[/red]")
                # Try to fetch bootstrap log for debugging
                log_content = fetch_bootstrap_log(s3, bucket)
                if log_content:
                    console.print("\n[yellow]Bootstrap log (last 20 lines):[/yellow]")
                    for line in log_content.strip().splitlines()[-20:]:
                        console.print(f"  {line}")
                return status_counts
        except Exception:
            pass

        time.sleep(POLL_INTERVAL_MONITOR)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def handle_sigint(signum, frame):
    global _active_pod_id
    console.print("\n\n[yellow]Interrupted![/yellow]")

    if _active_pod_id:
        console.print(f"Pod {_active_pod_id} is still running.")
        console.print("  t = Terminate pod")
        console.print("  l = Leave running (re-attach with --pod-id)")
        console.print("  c = Continue monitoring")

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
                console.print(f"[red]Failed: {e}[/red]")
            sys.exit(0)
        elif choice == "l":
            console.print("Pod left running. Re-attach:")
            console.print(f"  python 03_provision_simple.py --pod-id {_active_pod_id}")
            sys.exit(0)
        else:
            return
    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# Test file selection
# ---------------------------------------------------------------------------


def get_test_keys(s3, bucket: str) -> list[str]:
    """Pick 3 test files from the manifest: short, medium, long."""
    resp = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)
    manifest = json.loads(resp["Body"].read().decode("utf-8"))

    pending = [f for f in manifest["files"] if f["status"] == "pending"]
    if len(pending) < 3:
        return [f["key"] for f in pending]

    # Sort by file size as proxy for duration
    pending.sort(key=lambda f: f.get("size_bytes", 0))

    n = len(pending)
    # Short (~35MB), medium (50th pct), long (largest)
    short_idx = min(range(n), key=lambda i: abs(pending[i].get("size_bytes", 0) - 35_000_000))
    med_idx = n // 2
    long_idx = n - 1
    if med_idx == short_idx:
        med_idx = min(med_idx + 1, n - 1)
    if long_idx in (short_idx, med_idx):
        long_idx = max(long_idx - 1, 0)
    indices = [short_idx, med_idx, long_idx]
    selected = [pending[i]["key"] for i in indices]

    for key in selected:
        entry = next(f for f in pending if f["key"] == key)
        size_mb = entry.get("size_bytes", 0) / 1e6
        console.print(f"  Test file: {key.split('/')[-1]} ({size_mb:.1f}MB)")

    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU RunPod provisioner with R2 bootstrap",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Proof-of-concept: 3 files only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without creating pod",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging to console",
    )
    parser.add_argument(
        "--pod-id",
        type=str,
        default=None,
        help="Monitor an existing pod (skip provisioning)",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=None,
        help="Override GPU type (e.g. 'NVIDIA L40S')",
    )
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        help="Don't terminate pod on completion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="WhisperX batch size (default: 6)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    signal.signal(signal.SIGINT, handle_sigint)

    console.print(Panel("[bold]KSD Transcription — Single-GPU Provisioner[/bold]"))

    # Load config
    config = load_config()
    bucket = config["R2_BUCKET_NAME"]

    # Check R2
    s3 = create_s3_client(config)
    try:
        s3.head_bucket(Bucket=bucket)
        console.print(f"  [green]R2 bucket:[/green] {bucket}")
    except ClientError as e:
        console.print(f"[red]Cannot access R2 bucket: {e}[/red]")
        return 1

    # Check manifest
    status_counts = fetch_manifest_status(s3, bucket)
    if "error" in status_counts and not args.dry_run and not args.pod_id:
        console.print("[red]No manifest.json in R2. Run 01_upload.py first.[/red]")
        return 1

    total = status_counts.get("_total", 0)
    pending = status_counts.get("pending", 0)
    complete = status_counts.get("complete", 0)
    console.print(f"  [green]Manifest:[/green] {total} files ({pending} pending, {complete} complete)")

    if pending == 0 and not args.pod_id and not args.dry_run:
        console.print("[yellow]No pending files.[/yellow]")
        return 0

    # If just monitoring an existing pod, skip everything else
    if args.pod_id:
        console.print(f"\n  Monitoring pod: {args.pod_id}")
        global _active_pod_id
        _active_pod_id = args.pod_id
        final = monitor_manifest(s3, bucket, args.pod_id)
        _print_summary(final)
        return 0

    # Select GPU
    gpu_type_id, bid = select_gpu(args.gpu_type)
    console.print(f"  [green]GPU:[/green] {gpu_type_id} @ ${bid:.2f}/hr spot")

    # Test mode: select test keys
    test_keys = None
    if args.test:
        console.print("\n[bold]Test mode — selecting 3 test files...[/bold]")
        test_keys = get_test_keys(s3, bucket)
        if not test_keys:
            console.print("[yellow]No pending files for testing[/yellow]")
            return 0

    # Upload worker script to R2
    console.print("\n[bold]Preparing deployment...[/bold]")
    upload_worker_to_r2(s3, bucket)

    # Build bootstrap
    bootstrap_script = build_bootstrap(
        test_keys=test_keys,
        batch_size=args.batch_size,
        no_diarize=args.no_diarize,
    )
    docker_args = encode_bootstrap(bootstrap_script)
    logger.debug("Bootstrap script:\n%s", bootstrap_script)

    if args.dry_run:
        console.print("\n[yellow]DRY RUN — bootstrap script:[/yellow]")
        console.print(bootstrap_script)
        return 0

    # Select image based on GPU architecture
    image = IMAGE_PYTORCH_28 if gpu_type_id in BLACKWELL_GPUS else IMAGE_PYTORCH_24
    console.print(f"  [green]Image:[/green] {image}")

    # Create pod
    pod_id = create_pod(config, gpu_type_id, bid, docker_args, dry_run=False, image=image)
    if not pod_id:
        return 1

    _active_pod_id = pod_id

    # Wait for pod to start
    wait_for_running(pod_id)

    console.print("\n[bold]Pod is bootstrapping...[/bold]")
    console.print("  The pod is now installing dependencies and downloading models.")
    console.print("  First manifest changes should appear in ~5-10 minutes.")
    console.print("  You can debug via the RunPod web terminal if needed.")

    # Monitor via manifest
    final = monitor_manifest(s3, bucket, pod_id)

    # Terminate pod
    if not args.keep_pod:
        console.print(f"\n  Terminating pod {pod_id}...")
        try:
            runpod.terminate_pod(pod_id)
            console.print("  [green]Pod terminated[/green]")
            _active_pod_id = None
        except Exception as e:
            console.print(f"[red]Failed to terminate: {e}[/red]")
    else:
        console.print(f"\n  Pod {pod_id} left running (--keep-pod)")

    _print_summary(final)
    return 0


def _print_summary(status_counts: dict[str, int]) -> None:
    table = Table(title="Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Complete", f"[green]{status_counts.get('complete', 0)}[/green]")
    table.add_row("Failed", f"[red]{status_counts.get('failed', 0)}[/red]")
    table.add_row("Pending", str(status_counts.get("pending", 0)))
    table.add_row("Total", str(status_counts.get("_total", 0)))
    console.print()
    console.print(table)


if __name__ == "__main__":
    sys.exit(main())
