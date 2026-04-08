#!/usr/bin/env python3
"""Verify pyannote speaker diarization setup (CPU-only)."""

from __future__ import annotations

import math
import struct
import tempfile
import time
import wave
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Load HF_TOKEN from project .env files
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "transcription" / ".env",
]

for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(env_path)
        break

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    console.print("[red]HF_TOKEN not found in any .env file[/red]")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Step 1: Verify CPU-only torch
# ---------------------------------------------------------------------------

console.print("[bold]1. Checking PyTorch configuration[/bold]")
console.print(f"   torch version:    {torch.__version__}")
console.print(f"   cuda available:   {torch.cuda.is_available()}")
console.print(f"   cuda version:     {torch.version.cuda}")

assert not torch.cuda.is_available(), "CUDA should NOT be available (CPU-only build required)"
assert torch.version.cuda is None, "torch.version.cuda should be None (CPU-only build required)"

console.print("   [green]OK — CPU-only torch confirmed[/green]\n")


# ---------------------------------------------------------------------------
# Step 2: Load pyannote pipeline
# ---------------------------------------------------------------------------

console.print("[bold]2. Loading pyannote speaker-diarization-3.1 pipeline[/bold]")
t0 = time.perf_counter()

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
)
load_time = time.perf_counter() - t0
console.print(f"   Pipeline loaded in {load_time:.1f}s")
console.print(f"   pyannote.audio version: {__import__('pyannote.audio', fromlist=['__version__']).__version__}")
console.print("   [green]OK — pipeline loaded[/green]\n")


# ---------------------------------------------------------------------------
# Step 3: Generate synthetic audio and run diarization
# ---------------------------------------------------------------------------

console.print("[bold]3. Running diarization on 30s synthetic audio[/bold]")

SAMPLE_RATE = 16000
DURATION_S = 30
NUM_SAMPLES = SAMPLE_RATE * DURATION_S

# Generate a simple sine wave (440 Hz) as 16-bit PCM
samples = []
for i in range(NUM_SAMPLES):
    t = i / SAMPLE_RATE
    value = int(16000 * math.sin(2 * math.pi * 440 * t))
    samples.append(struct.pack("<h", value))

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(samples))

console.print(f"   Synthetic WAV: {tmp_path} ({DURATION_S}s, {SAMPLE_RATE}Hz mono)")

t0 = time.perf_counter()
diarization = pipeline(tmp_path)
infer_time = time.perf_counter() - t0

# Clean up
Path(tmp_path).unlink(missing_ok=True)

# Count speakers (pyannote 4.x returns DiarizeOutput dataclass)
annotation = diarization.speaker_diarization
speakers = set()
for turn, _, speaker in annotation.itertracks(yield_label=True):
    speakers.add(speaker)

console.print(f"   Diarization completed in {infer_time:.1f}s")
console.print(f"   Speakers detected: {len(speakers)}")
console.print("   [green]OK — pipeline runs without errors[/green]\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total_time = load_time + infer_time
console.print("[bold green]All checks passed![/bold green]")
console.print(f"   Pipeline load:  {load_time:.1f}s")
console.print(f"   Inference:      {infer_time:.1f}s")
console.print(f"   Total:          {total_time:.1f}s")
